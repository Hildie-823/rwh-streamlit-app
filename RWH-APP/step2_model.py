import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

df = pd.read_excel('clean_data.xlsx')
X = df.drop('BCR', axis=1)
y = df['BCR']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")

from sklearn.impute import SimpleImputer # 引入填补器
imputer = SimpleImputer(strategy='median')

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

X_train = pd.DataFrame(X_train_imputed, columns=X.columns)
X_test = pd.DataFrame(X_test_imputed, columns=X.columns)

models = {
    '随机森林': RandomForestRegressor(n_estimators=300, random_state=42),
    'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'MLP': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
}

results = {}
print("\n--- 5折交叉验证结果 ---")
for name, model in models.items():
    if name == 'MLP':
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        mae_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
    else:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        mae_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    
    results[name] = scores
    print(f"{name}: R² = {scores.mean():.4f} ± {scores.std():.4f}, MAE = {-mae_scores.mean():.4f}")

    plt.figure(figsize=(8, 6))
plt.boxplot(results.values(), tick_labels=results.keys())
plt.title('四模型交叉验证 R² 对比')
plt.ylabel('R² 得分')
plt.savefig('模型对比箱线图.png', dpi=300)
plt.close()

print("\n--- 开始随机森林超参数搜索 ---")
param_dist = {
    'n_estimators': [200, 300, 400, 500, 600, 700],
    'max_depth': [None, 8, 12, 16, 24],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8],
    'max_features': [0.4, 0.6, 0.8, 1.0]
}

rf = RandomForestRegressor(random_state=42)
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, 
                               n_iter=30, cv=5, scoring='r2', random_state=42, n_jobs=1)
rf_random.fit(X_train, y_train)

best_rf = rf_random.best_estimator_
print("最优参数:")
for k, v in rf_random.best_params_.items():
    print(f"  {k}: {v}")

y_pred = best_rf.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n测试集评估结果:")
print(f"R² = {r2:.4f}")
print(f"MAE = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print("提醒：测试集只能用一次！")

joblib.dump({'model': best_rf, 'features': X.columns.tolist()}, 'rf_bcr_model.joblib')
print("\n已保存模型 rf_bcr_model.joblib")
