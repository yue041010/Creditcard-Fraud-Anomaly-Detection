import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 加载CSV数据
df = pd.read_csv('ai_basic\data\your_file_with_random_decimal_column.csv')  # 确保路径正确

# 分离目标变量和输入特征
X = df.drop(columns=['fraud'])  # 输入特征
y = df['fraud']  # 目标变量

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化并训练孤独森林模型
clf = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
clf.fit(X_train)

# 对测试集进行预测
y_pred = clf.predict(X_test)

# 将预测结果中的1（正常）和-1（异常）调整为0和1，以便与目标变量对比
y_pred = [1 if x == -1 else 0 for x in y_pred]  # 1表示欺诈，0表示正常

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# 输出详细的分类报告
print(classification_report(y_test, y_pred))

# 使用某一个特征进行可视化
X_test_values = X_test.iloc[:, 7].values  # 使用第一个特征进行可视化

# 将异常点（欺诈）和正常点分开
anomaly_points = X_test_values[np.array(y_pred) == 1]
normal_points = X_test_values[np.array(y_pred) == 0]

# 绘制正常点的图
plt.figure(figsize=(8, 6))
plt.scatter(normal_points, np.zeros_like(normal_points), color='blue', label='Normal (0)')
plt.title('Normal Points')
plt.xlabel('Feature 1')
plt.ylabel('Density')
plt.legend()
plt.show()

# 绘制异常点的图
plt.figure(figsize=(8, 6))
plt.scatter(anomaly_points, np.zeros_like(anomaly_points), color='red', label='Fraud (1)')
plt.title('Anomaly Points')
plt.xlabel('Feature 1')
plt.ylabel('Density')
plt.legend()
plt.show()
