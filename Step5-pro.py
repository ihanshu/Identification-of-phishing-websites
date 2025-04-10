import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 加载数据集
data = pd.read_csv("url_features_final.csv")

# 假设数据集中的最后一列是标签列（label），其余是特征
X = data.iloc[:, :-1]  # 特征
y = data.iloc[:, -1]   # 标签

# 划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用随机森林模型来评估特征重要性
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 获取特征重要性
feature_importances = rf_model.feature_importances_

# 将特征重要性与特征名称对应起来
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# 打印特征重要性
print("Feature Importances:")
print(feature_importance_df)

# 选择重要特征（例如选择重要性大于某个阈值的特征）
threshold = 0.01  # 可以根据实际情况调整阈值
important_features = feature_importance_df[feature_importance_df['Importance'] > threshold]['Feature'].tolist()

# 使用重要特征重新划分数据集
X_train_important = X_train[important_features]
X_test_important = X_test[important_features]

# 使用重要特征重新训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_important, y_train)

# 保存改进后的模型
joblib.dump(model, "phishing_url_model_improved.pkl")

# 加载改进后的模型
model = joblib.load("phishing_url_model_improved.pkl")

# 使用改进后的模型预测测试集
y_pred = model.predict(X_test_important)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy with Important Features: {accuracy:.4f}")

# 打印详细的分类报告
print("Classification Report with Important Features:")
print(classification_report(y_test, y_pred))