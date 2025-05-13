import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import BertTokenizer

# ========== Step 1. 数据加载 ==========
df = pd.read_csv(r"D:\projects\BERT\帮助信息网络安全犯罪2_new.csv")

# ========== Step 2. 缺失值处理 ==========
df["案由"] = df["案由"].fillna("未知")
df["裁判依据"] = df["裁判依据"].fillna("缺失")
df["类型"] = df["类型"].fillna("未知")
df["地址"] = df["地址"].fillna("未知")
df["时间"] = pd.to_datetime(df["时间"], errors="coerce")  # 转为 datetime 格式
df = df.dropna(subset=["罚金"])  # 删除罚金为空的样本

# ========== Step 3. 刑期转换（月） ==========
def extract_months(term):
    if pd.isna(term):
        return None
    y = re.search(r"(\d+)年", term)
    m = re.search(r"(\d+)个月", term)
    return (int(y.group(1)) * 12 if y else 0) + (int(m.group(1)) if m else 0)

df["刑期（月）"] = df["刑期"].apply(extract_months)

# ========== Step 4. 刑期标签分类 ==========
def classify_term(months):
    if months is None:
        return None
    if months <= 12:
        return 0  # 轻刑
    elif months <= 36:
        return 1  # 中刑
    return 2      # 重刑

df["label"] = df["刑期（月）"].apply(classify_term)

# ========== Step 5. 文本拼接（案由 + 裁判结果） ==========
df["text"] = df["案由"].fillna('') + "。" + df["裁判结果"].fillna('')

# ========== Step 6. 删除建模无效数据 ==========
df = df.dropna(subset=["text", "label"])

# ========== Step 7. 数据集划分 ==========
train_df, test_df = train_test_split(
    df[["text", "label"]],
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# ========== Step 8. 分词器准备 ==========
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# ========== Step 9. 导出处理后的数据 ==========
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)
df.to_excel(os.path.join(output_dir, "processed_dataset.xlsx"), index=False)
print("✅ 已保存至 ./data/processed_dataset.xlsx")

from collections import Counter
print(Counter(df["label"]))
print("✅ 标签分布:", Counter(df["label"]))