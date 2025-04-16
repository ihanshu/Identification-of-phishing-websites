import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
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

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_important)
X_test_scaled = scaler.transform(X_test_important)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# 定义深度学习模型
class PhishingURLClassifier(nn.Module):
    def __init__(self, input_dim):
        super(PhishingURLClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # 输入层到隐藏层1
        self.fc2 = nn.Linear(64, 32)         # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(32, 16)         # 隐藏层2到隐藏层3
        self.fc4 = nn.Linear(16, 2)          # 隐藏层3到输出层（二分类问题）

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU 激活函数
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # 输出层，使用softmax激活函数
        return x

# 实例化模型
input_dim = X_train_important.shape[1]  # 输入维度（特征数）
model = PhishingURLClassifier(input_dim)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器[^1^]

# 训练模型
num_epochs = 100
batch_size = 32
for epoch in range(num_epochs):
    model.train()

    # 打乱数据
    indices = torch.randperm(X_train_tensor.size(0))
    X_train_tensor = X_train_tensor[indices]
    y_train_tensor = y_train_tensor[indices]

    # 按批次训练
    for i in range(0, X_train_tensor.size(0), batch_size):
        # 获取当前批次数据
        X_batch = X_train_tensor[i:i+batch_size]
        y_batch = y_train_tensor[i:i+batch_size]

        # 前向传播
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # 反向传播和优化
        optimizer.zero_grad()  # 清除之前的梯度
        loss.backward()        # 反向传播
        optimizer.step()       # 更新模型参数

    # 每隔一定周期打印训练损失
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 保存改进后的模型
torch.save(model.state_dict(), "phishing_url_model_improved.pth")

# 加载改进后的模型
model.load_state_dict(torch.load("phishing_url_model_improved.pth"))

# 验证模型
model.eval()
with torch.no_grad():  # 不需要计算梯度
    val_outputs = model(X_test_tensor)
    _, predicted = torch.max(val_outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f'Test Accuracy: {accuracy:.4f}')
