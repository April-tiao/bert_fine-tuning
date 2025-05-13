import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns  

from transformers import TrainingArguments

from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ========== Step 1. 数据加载 ==========
df = pd.read_csv(r"D:\projects\BERT\帮助信息网络安全犯罪2_new.csv")

# ========== Step 2. 缺失值处理 ==========
df["案由"] = df["案由"].fillna("未知")
df["裁判依据"] = df["裁判依据"].fillna("缺失")
df["类型"] = df["类型"].fillna("未知")
df["地址"] = df["地址"].fillna("未知")
df["时间"] = pd.to_datetime(df["时间"], errors="coerce")
df = df.dropna(subset=["罚金"])

# ========== Step 3. 刑期转换（月） ==========
def extract_months(term):
    if pd.isna(term):
        return None
    y = re.search(r"(\d+)年", term)
    m = re.search(r"(\d+)个月", term)
    return (int(y.group(1)) * 12 if y else 0) + (int(m.group(1)) if m else 0)

df["刑期（月）"] = df["刑期"].apply(extract_months)

# ========== Step 4. 标签分类 ==========
def classify_term(months):
    if months is None:
        return None
    if months <= 12:
        return 0
    elif months <= 36:
        return 1
    return 2

df["label"] = df["刑期（月）"].apply(classify_term)

# ========== Step 5. 文本拼接 ==========
df["text"] = df["案由"].fillna('') + "。" + df["裁判结果"].fillna('')
df = df.dropna(subset=["text", "label"])

# ========== Step 6. 划分数据集 ==========
train_df, test_df = train_test_split(
    df[["text", "label"]],
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# ========== Step 7. 分词器 ==========
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# ========== Step 8. 自定义 FocalLoss ==========
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        ce_loss = F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean() if self.reduction == "mean" else loss.sum()

# ========== Step 9. 自定义 Trainer ==========
class FocalLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = FocalLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ========== Step 10. 模型与训练参数 ==========
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# training_args = TrainingArguments(
#     output_dir="./results",
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     gradient_accumulation_steps=1,
#     num_train_epochs=5,
#     learning_rate=2e-5,
#     weight_decay=0.01,
#     # evaluation_strategy="epoch",
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     logging_strategy="epoch",
#     lr_scheduler_type="linear",
#     warmup_steps=200,
#     load_best_model_at_end=True,
#     metric_for_best_model="f1",
#     save_total_limit=2,
#     fp16=True,  # 需要安装 NVIDIA apex 或 PyTorch AMP 支持
#     report_to="none"  # 关闭 wandb 等远程日志
# )
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
    num_train_epochs=20,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    lr_scheduler_type="cosine",  # 更平滑
    warmup_steps=50,  # 数据量小，可减少 warmup
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2,
    fp16=True,
    report_to="none"
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

trainer = FocalLossTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ========== Step 11. 模型训练 ==========
train_result = trainer.train()
trainer.save_model()


# ========== Step 12. 绘制训练与验证 Loss 曲线 ==========
logs = trainer.state.log_history
losses = [log["loss"] for log in logs if "loss" in log]
eval_losses = [log["eval_loss"] for log in logs if "eval_loss" in log]
steps = [log["step"] for log in logs if "loss" in log]

plt.figure(figsize=(8, 6))
plt.plot(steps, losses, label="Train Loss", color="royalblue", linewidth=2)
if eval_losses:
    eval_steps = [log["step"] for log in logs if "eval_loss" in log]
    plt.plot(eval_steps, eval_losses, label="Eval Loss", color="darkorange", linewidth=2)

plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ========== Step 13. 模型评估 ==========
predictions = trainer.predict(tokenized_test)
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, digits=4))

# ========== Step 13. 绘制混淆矩阵热力图 ==========
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap="Blues",      # 越深颜色代表样本数越多
    cbar=True,
    square=True,
    linewidths=0.5,
    linecolor="gray"
)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix (Focal Loss + BERT)")
plt.tight_layout()
plt.show()

# ========== Step 14. 保存预处理数据 ==========
os.makedirs("data", exist_ok=True)
df.to_excel("data/processed_dataset.xlsx", index=False)
print("✅ 已保存数据至 ./data/processed_dataset.xlsx")
print("✅ 标签分布:", Counter(df["label"]))
