# kmeans_cluster.py
# -----------------------------------------------
# 聚类刑期（月）+ log_罚金 + BERT-PCA 语义主成分
# -----------------------------------------------
import os, re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import BertTokenizer, BertModel

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥  running on: {device}")

# ---------- 1. 数据读取 ----------
CSV_PATH = r"D:\projects\BERT\帮助信息网络安全犯罪2_new.csv"  # ← 改成自己的路径
df = pd.read_csv(CSV_PATH)

# ---------- 2. 字段预处理 ----------
def extract_months(term: str) -> float:
    if pd.isna(term):
        return np.nan
    y = re.search(r"(\d+)年", term)
    m = re.search(r"(\d+)个月", term)
    return (int(y.group(1))*12 if y else 0) + (int(m.group(1)) if m else 0)

df["刑期（月）"] = df["刑期"].apply(extract_months)
df["log_罚金"]   = np.log1p(df["罚金"])
df.dropna(subset=["刑期（月）", "log_罚金"], inplace=True)

df["案由"]   = df["案由"].fillna("未知")
df["裁判结果"] = df["裁判结果"].fillna("")
df["text"]   = df["案由"] + "。" + df["裁判结果"]

# ---------- 3. BERT CLS 向量 ----------
class TextDS(TorchDataset):
    def __init__(self, texts): self.texts = texts
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return self.texts[i]

model_name = "bert-base-chinese"
tokenizer   = BertTokenizer.from_pretrained(model_name)
bert_model  = BertModel.from_pretrained(model_name).to(device).eval()

def collate_fn(batch):
    enc = tokenizer(batch, padding=True, truncation=True,
                    max_length=512, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}

loader = DataLoader(TextDS(df["text"].tolist()),
                    batch_size=32, shuffle=False, collate_fn=collate_fn)

embeddings = []
with torch.no_grad():
    for batch in tqdm(loader, desc="BERT embedding"):
        out = bert_model(**batch).last_hidden_state[:, 0, :]   # CLS
        embeddings.append(out.cpu())
embeddings = torch.cat(embeddings).numpy()                    # (N, 768)

# ---------- 4. PCA → 2 维 ----------
pca = PCA(n_components=2, random_state=42)
pca_vec = pca.fit_transform(embeddings)                        # (N, 2)
df["emb_pca1"] = pca_vec[:, 0]
df["emb_pca2"] = pca_vec[:, 1]

# ---------- 5. 聚类准备 ----------
feature_cols = ["刑期（月）", "log_罚金", "emb_pca1", "emb_pca2"]
X = df[feature_cols].values
X = StandardScaler().fit_transform(X)                          # 标准化

# ---------- 6. 生成过程表格 ----------
process_table = pd.DataFrame({
    "步骤": [1, 2, 3, 4, 5],
    "关键操作": ["特征选取", "标准化", "K-means 训练", "评估 K", "结果输出"],
    "细节": [", ".join(feature_cols),
             "StandardScaler 均值0方差1",
             "KMeans(n_init='auto', random_state=42)",
             "inertia_ / silhouette_score",
             "Excel+散点图"]
})
print("\n🗒  过程表格")
print(process_table.to_markdown(index=False))

# ---------- 7. K-means (k=3,4) ----------
k_list = [3, 4]
summary_tables = {}

for k in k_list:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    df[f"cluster{k}"] = labels
    sil = silhouette_score(X, labels)
    print(f"✅  k={k}  inertia={km.inertia_:.1f}  silhouette={sil:.3f}")

    # 汇总统计
    tbl = (df.groupby(f"cluster{k}")[feature_cols]
           .agg(["count", "mean", "std"])
           .round(2))
    summary_tables[k] = tbl
    print(f"\n📊  k={k} 结果表格")
    print(tbl.to_markdown())

    # 散点图
    plt.figure(figsize=(6,5))
    sns.scatterplot(x="emb_pca1", y="emb_pca2",
                    hue=f"cluster{k}",
                    palette=sns.color_palette("Set2", k),
                    data=df, s=40, edgecolor="k")
    plt.title(f"K-means 聚类 (k={k}) in PCA Space")
    plt.xlabel("emb_pca1")
    plt.ylabel("emb_pca2")
    plt.legend(title="cluster", loc="best")
    plt.tight_layout()
    plt.savefig(f"cluster_scatter_k{k}.png", dpi=300)
    plt.show()

# ---------- 8. 保存 Excel ----------
os.makedirs("cluster_output", exist_ok=True)
with pd.ExcelWriter("cluster_output/cluster_summary.xlsx") as writer:
    process_table.to_excel(writer, sheet_name="process", index=False)
    for k, tbl in summary_tables.items():
        tbl.to_excel(writer, sheet_name=f"k{k}")

print("\n🎉  全部完成！已生成：")
print("    · cluster_output/cluster_summary.xlsx")
print("    · cluster_scatter_k3.png / cluster_scatter_k4.png")
