
import os, re, math
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset

import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.decomposition import PCA
from transformers import BertTokenizer, BertModel

# ---------------- 全局设置 ----------------
plt.rcParams["font.family"] = "SimHei"          # 中文
plt.rcParams["axes.unicode_minus"] = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥  running on: {device}")

# ---------------- 读取 & 预处理 ----------------
df = pd.read_csv(r"D:\projects\BERT\帮助信息网络安全犯罪2_new.csv")

# 缺失值处理
df["案由"]   = df["案由"].fillna("未知")
df["裁判依据"] = df["裁判依据"].fillna("缺失")
df["类型"]   = df["类型"].fillna("未知")
df["地址"]   = df["地址"].fillna("未知")
df["时间"]   = pd.to_datetime(df["时间"], errors="coerce")
df          = df.dropna(subset=["罚金"])

# 刑期（月）
def extract_months(term):
    if pd.isna(term):
        return np.nan
    y = re.search(r"(\d+)年", term)
    m = re.search(r"(\d+)个月", term)
    return (int(y.group(1)) * 12 if y else 0) + (int(m.group(1)) if m else 0)

df["刑期（月）"] = df["刑期"].apply(extract_months)
df["log_罚金"]   = np.log1p(df["罚金"])         # 长尾 → 对数
df = df.dropna(subset=["刑期（月）"])            # 少量缺失可直接剔除

# 拼接文本
df["text"] = df["案由"].fillna("") + "。" + df["裁判结果"].fillna("")

# ---------------- 文本 BERT 向量 ----------------
model_name = "bert-base-chinese"
tokenizer   = BertTokenizer.from_pretrained(model_name)
bert_model  = BertModel.from_pretrained(model_name).to(device).eval()

class TextDataset(TorchDataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]

def collate_fn(batch):
    enc = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    return {k: v.to(device) for k, v in enc.items()}

loader = DataLoader(
    TextDataset(df["text"].tolist()),
    batch_size=32,              # ↔ GPU 显存
    shuffle=False,
    collate_fn=collate_fn
)

embeddings = []
with torch.no_grad():
    for batch in tqdm(loader, desc="BERT embedding"):
        out = bert_model(**batch).last_hidden_state[:, 0, :]   # CLS
        embeddings.append(out.cpu())
embeddings = torch.cat(embeddings).numpy()                    # (N, 768)

emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
df[emb_cols] = pd.DataFrame(embeddings, index=df.index)

# ---------------- Spearman 相关 ----------------
numeric_cols = ["刑期（月）", "罚金", "log_罚金"]

# 1) 3 × 768 子矩阵
corr_large = (
    df[numeric_cols + emb_cols]
    .corr(method="spearman")
    .loc[numeric_cols, emb_cols]
    .abs()
)

# 2) PCA 前 2 维 → 5×5 小矩阵
pca = PCA(n_components=2, random_state=42)
pca_vec = pca.fit_transform(embeddings)          # (N, 2)
df["emb_pca1"] = pca_vec[:, 0]
df["emb_pca2"] = pca_vec[:, 1]
small_cols = numeric_cols + ["emb_pca1", "emb_pca2"]
corr_small = df[small_cols].corr(method="spearman")

# ---------------- 绘图 ----------------
def plot_heat(data, xticks, yticks, title, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(xticks)))
    ax.set_xticklabels(xticks, rotation=45, ha="right")
    ax.set_yticks(range(len(yticks)))
    ax.set_yticklabels(yticks)
    # 数值标注（小图才写）
    if data.shape[0] <= 10 and data.shape[1] <= 10:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, label="|Spearman ρ|")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

plot_heat(
    corr_large.values,
    xticks=[],                         # 768 维太多
    yticks=numeric_cols,
    title="Numeric ↔ BERT-CLS 相关 (|ρ|)",
    figsize=(28, 4)
)

plot_heat(
    corr_small.abs().values,
    xticks=small_cols,
    yticks=small_cols,
    title="Spearman 相关矩阵 (含 BERT-PCA)",
    figsize=(6, 5)
)
