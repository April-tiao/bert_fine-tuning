import pandas as pd

# 读取数据
df = pd.read_csv(r"D:\projects\BERT\帮助信息网络安全犯罪2_new.csv", encoding="utf-8")  # 注意根据需要修改编码

# 1. 查看行列数和字段名
print("➡️ 数据维度 (行, 列):", df.shape)
print("➡️ 字段名列表:")
print(df.columns.tolist())

# 2. 每列数据类型和非空数量
print("\n➡️ 字段类型和非空值数量:")
print(df.info())

# 3. 缺失值统计（按列）
print("\n➡️ 缺失值统计:")
print(df.isnull().sum())

# 4. 数值型列的描述性统计
print("\n➡️ 数值型统计信息:")
print(df.describe())

# 5. 非数值型列的唯一值个数
print("\n➡️ 非数值型字段唯一值统计:")
print(df.select_dtypes(include='object').nunique())

# 6. 查看前几行
print("\n➡️ 数据预览:")
print(df.head(5))

exit()

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
import torch

# ========== Step 1. 数据加载 ==========
df = pd.read_csv(r"D:\projects\BERT\帮助信息网络安全犯罪2_new.csv")

# ========== Step 2. 文本清洗 + 标签构建 ==========
def extract_months(term):
    if pd.isna(term):
        return None
    y = re.search(r"(\d+)年", term)
    m = re.search(r"(\d+)个月", term)
    return (int(y.group(1)) * 12 if y else 0) + (int(m.group(1)) if m else 0)

def classify_term(months):
    if months is None:
        return None
    if months <= 12:
        return 0
    elif months <= 36:
        return 1
    return 2

df["刑期（月）"] = df["刑期"].apply(extract_months)
df["label"] = df["刑期（月）"].apply(classify_term)
df["text"] = df["案由"].fillna('') + "。" + df["裁判结果"].fillna('')
df = df.dropna(subset=["text", "label"])

# ========== Step 3. 数据集划分 ==========
train_df, test_df = train_test_split(df[["text", "label"]], test_size=0.2, random_state=42, stratify=df["label"])
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# ========== Step 4. 分词器 ==========
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=512)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# ========== Step 5. 模型准备 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.to(device)

# ========== Step 6. 训练参数 ==========
args = TrainingArguments(
    output_dir="./bert刑期分类",
    eval_strategy="epoch",            # 修改此处
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",  # 关闭中间日志系统
)

# ========== Step 7. Trainer ==========
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=lambda p: {
        "accuracy": (p.predictions.argmax(axis=-1) == p.label_ids).astype(float).mean().item()
    },
)

# ========== Step 8. 训练 ==========
trainer.train()

# ========== Step 9. 模型评估 ==========
with torch.no_grad():
    # preds = trainer.predict(tokenized_test)
    preds = trainer.predict(tokenized_test, metric_key_prefix="test")
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(axis=-1)

print(classification_report(y_true, y_pred, target_names=["轻刑", "中刑", "重刑"]))
