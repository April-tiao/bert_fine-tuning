import pandas as pd
import re
import os

# 模拟加载数据
data = {
    "案由": ["帮助信息网络犯罪活动", "帮助信息网络犯罪活动", None, "帮助信息网络犯罪活动"],
    "裁判结果": [
        "判处有期徒刑六个月，并处罚金人民币五千元。",
        "判处被告人有期徒刑一年六个月，并处罚金人民币五千元。",
        "判处被告人有期徒刑三年。",
        "判处被告人有期徒刑十个月，并处罚金人民币二千元。"
    ],
    "刑期": ["6个月", "1年6个月", "3年", "10个月"]
}
df = pd.DataFrame(data)

# 提取月份
def extract_months(term):
    if pd.isna(term):
        return None
    y = re.search(r"(\d+)年", term)
    m = re.search(r"(\d+)个月", term)
    return (int(y.group(1)) * 12 if y else 0) + (int(m.group(1)) if m else 0)

df["刑期（月）"] = df["刑期"].apply(extract_months)

# 分类标签
def classify_term(months):
    if months is None:
        return None
    if months <= 12:
        return 0
    elif months <= 36:
        return 1
    return 2

df["label"] = df["刑期（月）"].apply(classify_term)

# 拼接文本
df["text"] = df["案由"].fillna('') + "。" + df["裁判结果"].fillna('')

# import ace_tools as tools; tools.display_dataframe_to_user(name="刑期处理示例表", dataframe=df)

print(df.to_string(index=False)) 


# 设置为项目中的 data 文件夹
target_dir = "data"
os.makedirs(target_dir, exist_ok=True)

# 拼接文件路径
target_path = os.path.join(target_dir, "processed_penalty_data.xlsx")

# 保存 Excel 文件
# df.to_excel(target_path, index=False)
# print(f"文件已保存至: {target_path}")
