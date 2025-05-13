# BERT-Based Sentence Classification for Criminal Judgment Analysis

本项目旨在使用中文 BERT 模型对刑事判决文书中的案由与裁判结果进行分类，预测对应的刑期等级。特别地，本项目使用了 **Focal Loss** 来处理类别不平衡问题，并使用 Hugging Face 的 `transformers` 库进行训练和评估。

## 📁 项目结构

```
.
├── data/                            # 存放处理后的数据
│   └── processed_dataset.xlsx
├── 帮助信息网络安全犯罪2_new.csv   # 原始数据文件（需手动放置）
├── main.py                          # 主训练脚本
├── results/                         # 模型输出目录
└── README.md
```

## 🚀 项目亮点

- 使用 `bert-base-chinese` 模型进行文本分类
- 支持自定义的 **Focal Loss**（用于处理类别不平衡）
- 模型训练、评估与可视化一体化实现
- 分类目标为刑期长度（分为三类）
- 自动绘制 Loss 曲线和混淆矩阵热力图

## 🛠️ 环境依赖

```bash
pip install pandas numpy torch matplotlib scikit-learn seaborn datasets transformers
```

或使用如下 `conda` 环境：

```bash
conda create -n bert-law python=3.9
conda activate bert-law
pip install -r requirements.txt  # 可选，需你导出 requirements
```

## 📊 数据说明

数据文件：`帮助信息网络安全犯罪2_new.csv`，包含字段：

- `案由`、`裁判依据`、`裁判结果`：案件内容文本
- `刑期`：原始中文刑期（如“2年3个月”）
- `罚金`、`地址`、`时间` 等字段

标签划分：

- 刑期 ≤ 12 个月 → 类别 0
- 12 < 刑期 ≤ 36 个月 → 类别 1
- 刑期 > 36 个月 → 类别 2

## 🔍 运行说明

1. 将数据文件放置于指定路径：

   ```bash
   D:\projects\BERT\帮助信息网络安全犯罪2_new.csv
   ```

   或修改脚本中的路径。

2. 运行主程序：

```bash
python main.py
```

训练过程包括：

- 数据预处理（缺失值填充、刑期提取与分类）
- 数据集划分与分词
- 自定义 Trainer 训练 BERT 模型
- 自动保存模型与日志
- 可视化训练 Loss 和混淆矩阵

## 📈 模型评估指标

- **Accuracy**
- **F1-score（加权）**
- 分类报告输出
- 混淆矩阵热力图展示模型表现

## 📦 模型保存

训练结束后，模型将保存在 `./results` 文件夹中，训练数据将被保存为 `data/processed_dataset.xlsx`。

## 📌 注意事项

- 默认使用 GPU 训练（开启 `fp16` 混合精度）
- 如无 NVIDIA GPU，请将 `fp16=True` 改为 `False`
- 数据较少时建议减少 `warmup_steps`，如已设置为 50

  
