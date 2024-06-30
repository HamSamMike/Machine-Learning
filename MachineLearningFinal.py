import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 读取数据
file_path = '考查-单车.csv'
data = pd.read_csv(file_path)

# 数据预处理
data['Start date'] = pd.to_datetime(data['Start date'])
data['End date'] = pd.to_datetime(data['End date'])
data['Ride Duration (min)'] = (data['End date'] - data['Start date']).dt.total_seconds() / 60
data = data[data['Ride Duration (min)'] > 0]
data.dropna(inplace=True)

# 将Total duration (ms)转换为分钟
data['Total duration (min)'] = data['Total duration (ms)'] / 60000

# 设定合理的骑行时间范围，过滤掉超过240分钟的记录
reasonable_data = data[data['Total duration (min)'] <= 240]

# 去除Total duration的极值，设定合理范围（例如0到240分钟）
filtered_data = reasonable_data[reasonable_data['Total duration (min)'] <= 240]

# 创建Total duration的直方图
plt.figure(figsize=(10, 6))
plt.hist(filtered_data['Total duration (min)'], bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('Total duration Distribution (Filtered)')
plt.xlabel('Total duration (minutes)')
plt.ylabel('Frequency')
plt.show()

# 将Bike model转换为数值型变量（例如0表示CLASSIC，1表示PBSC_EBIKE）
filtered_data['Bike model encoded'] = filtered_data['Bike model'].apply(lambda x: 0 if x == 'CLASSIC' else 1)

# 使用seaborn绘制箱线图
plt.figure(figsize=(12, 8))
sns.boxplot(x='Bike model', y='Total duration (min)', data=filtered_data)
plt.title('Boxplot of Total duration by Bike model')
plt.xlabel('Bike model')
plt.ylabel('Total duration (minutes)')
plt.show()

# 对Start station和End station进行编码
filtered_data['Start station encoded'] = filtered_data['Start station number'].astype('category').cat.codes
filtered_data['End station encoded'] = filtered_data['End station number'].astype('category').cat.codes

# 选择骑行次数最多的前20个Start station
top_start_stations = filtered_data['Start station number'].value_counts().nlargest(20).index
top_start_data = filtered_data[filtered_data['Start station number'].isin(top_start_stations)]

# 输出前20个Start station
print("前20个骑行次数最多的起始站点:")
print(top_start_stations)

# 分析Start station对Total duration的影响
plt.figure(figsize=(12, 8))
sns.boxplot(x='Start station number', y='Total duration (min)', data=top_start_data)
plt.title('Total duration by Top 20 Start stations')
plt.xlabel('Start station')
plt.ylabel('Total duration (minutes)')
plt.xticks(rotation=90)
plt.show()

# 选择骑行次数最多的前20个End station
top_end_stations = filtered_data['End station number'].value_counts().nlargest(20).index
top_end_data = filtered_data[filtered_data['End station number'].isin(top_end_stations)]

# 输出前20个End station
print("前20个骑行次数最多的结束站点:")
print(top_end_stations)

# 分析End station对Total duration的影响
plt.figure(figsize=(12, 8))
sns.boxplot(x='End station number', y='Total duration (min)', data=top_end_data)
plt.title('Total duration by Top 20 End stations')
plt.xlabel('End station')
plt.ylabel('Total duration (minutes)')
plt.xticks(rotation=90)
plt.show()

# 提取小时信息
filtered_data['Start hour'] = filtered_data['Start date'].dt.hour

# 统计每小时的骑行频率
hourly_counts = filtered_data['Start hour'].value_counts().sort_index()

# 绘制柱状图
plt.figure(figsize=(12, 8))
sns.barplot(x=hourly_counts.index, y=hourly_counts.values, palette='viridis')
plt.title('Hourly Ride Frequency')
plt.xlabel('Hour of Day')
plt.ylabel('Ride Count')
plt.xticks(range(24))
plt.show()

# 提取星期几信息
filtered_data['Start day'] = filtered_data['Start date'].dt.dayofweek

# 统计每一天的骑行频率
daily_counts = filtered_data['Start day'].value_counts().sort_index()

# 绘制柱状图
plt.figure(figsize=(12, 8))
sns.barplot(x=daily_counts.index, y=daily_counts.values, palette='viridis')
plt.title('Daily Ride Frequency')
plt.xlabel('Day of Week')
plt.ylabel('Ride Count')
plt.xticks(ticks=range(7), labels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.show()

# 提取小时和星期几信息
data['Start hour'] = data['Start date'].dt.hour
data['Start day'] = data['Start date'].dt.dayofweek

# 对类别特征进行编码
label_encoder = LabelEncoder()
data['Start station encoded'] = label_encoder.fit_transform(data['Start station number'])
data['End station encoded'] = label_encoder.fit_transform(data['End station number'])

# 按小时聚合数据，计算每小时的骑行次数
hourly_data = data.groupby(['Start day', 'Start hour']).size().reset_index(name='Ride count')

# 定义特征和目标变量
features = ['Start day', 'Start hour']
X = hourly_data[features]
y = hourly_data['Ride count']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 随机森林回归模型超参数调优（缩小参数网格）
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=2, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_

# 评估模型
y_pred_rf = best_rf_model.predict(X_test)
print("随机森林回归模型（调优后）:")
print(f"均方误差: {mean_squared_error(y_test, y_pred_rf)}")
print(f"R2评分: {r2_score(y_test, y_pred_rf)}")

# 可视化预测结果与实际值的比较
plt.figure(figsize=(12, 8))
plt.scatter(y_test, y_pred_rf, alpha=0.3)
plt.plot([0, max(y_test)], [0, max(y_test)], '--r', linewidth=2)
plt.xlabel('Actual Ride Count')
plt.ylabel('Predicted Ride Count')
plt.title('Actual vs Predicted Ride Count')
plt.show()