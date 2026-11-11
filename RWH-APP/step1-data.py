import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sys
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print(f"Python 版本: {sys.version}")
print(f"Pandas 版本: {pd.__version__}")
print(f"Scikit-learn 版本: {sklearn.__version__}")

df = pd.read_excel('Sample_dataset.xlsx', sheet_name='Sheet1')
print(f"原始数据行列数: {df.shape}")

features = [
    'Water Tariff (USD/m3)', 'Rainfall (mm/year)', 'Coefficient of Variation', 
    'Catchment Area (m2)', 'Total Built Area (m2)', 'Demand (m3/d)', 
    'Number of Occupants', 'Building Height (m)', 'Tank Sizes (m3)', 
    'Project Lifetime (Year)', 'Energy Cost (USD/kWh)', 'Discount Rate (%)'
]
target = 'BCR'
cols_to_keep = features + [target]

try:
    df_clean = df[cols_to_keep].copy()
except KeyError as e:
    print(f"列名报错: 找不到列 {e}，请检查Excel表头！")
    # 如果报错，打印出所有列名供你核对
    print("当前Excel的所有列名：", df.columns.tolist())
    sys.exit()

df_clean.dropna(subset=[target], inplace=True)
for col in features:
    df_clean[col].fillna(df_clean[col].median(), inplace=True)
df_clean[target] = df_clean[target].apply(lambda x: 0 if x < 0 else x)

print(f"清洗后数据行列数: {df_clean.shape}")
print(f"剩余缺失值总数: {df_clean.isnull().sum().sum()}")

corr = df_clean.corr()[target].drop(target).abs().sort_values(ascending=False)
print("\n与 BCR 相关系数绝对值前5的变量：")
print(corr.head(5))

cn_cols = ['水价', '降雨量', '变异系数', '汇水面积', '建筑面积', '需水量', 
           '人数', '楼高', '水箱容积', '项目寿命', '电价', '贴现率', 'BCR']
df_clean.columns = cn_cols
df_clean.to_excel('clean_data.xlsx', index=False)
print("\n已保存 clean_data.xlsx")

fig, axes = plt.subplots(3, 4, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(cn_cols[:-1]):
    sns.violinplot(y=df_clean[col], ax=axes[i], color='skyblue')
    axes[i].set_title(f'{col} 分布')
plt.tight_layout()
plt.savefig('小提琴图.png', dpi=300)
plt.show()

plt.figure(figsize=(12, 10))
sns.heatmap(df_clean.corr(), annot=True, cmap='Blues', fmt='.2f')
plt.title('特征与 BCR 相关性热图')
plt.tight_layout()
plt.savefig('相关性热图.png', dpi=300)
plt.close()
print("特征图已生成！")
