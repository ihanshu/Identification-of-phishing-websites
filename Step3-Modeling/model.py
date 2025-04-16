import pandas as pd
from sklearn.model_selection import train_test_split

# 加载数据集
data = pd.read_csv("url_features_final.csv")

# 假设数据集中的最后一列是标签列（label），其余是特征
X = data.iloc[:, :-1]  # 特征
y = data.iloc[:, -1]   # 标签

# 划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# 定义模型
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "KNN": KNeighborsClassifier()
}

# 交叉验证评估模型
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name} Cross-Validation Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")


from sklearn.ensemble import RandomForestClassifier
import joblib

# 选择随机森林模型进行训练
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 保存模型
joblib.dump(model, "phishing_url_model.pkl")

from sklearn.metrics import accuracy_score, classification_report

# 加载模型
model = joblib.load("phishing_url_model.pkl")

# 预测测试集
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# 打印详细的分类报告
print("Classification Report:")
print(classification_report(y_test, y_pred))

import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 加载数据集
data = pd.read_csv("url_features_final.csv")

# 假设数据集中的最后一列是标签列（label），其余是特征
X = data.iloc[:, :-1]  # 特征
y = data.iloc[:, -1]   # 标签

# 划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用 SelectKBest 选择最重要的 K 个特征
k = 8  # 选择 8 个最重要的特征
selector = SelectKBest(chi2, k=k)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# 获取选择的特征名称
selected_features = X.columns[selector.get_support()]
print("Selected Features:", selected_features)

# 重新训练模型
model = RandomForestClassifier()
model.fit(X_train_selected, y_train)

# 评估模型
y_pred = model.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")


# 保存选择的特征
selected_features.to_csv("selected_features.csv", index=False)

# 保存模型
joblib.dump(model, "phishing_url_model_selected.pkl")