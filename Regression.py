# 自整理使用的sklearn实现的机器学习回归算法
# 以下代码是针对我的数据进行编写，如需使用请替换数据

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA


file_path = r'D:\Phyon\Project\bys\train_data.xlsx'  # 请替换为自己的数据
data = pd.read_excel(file_path,header=0)  # header为表头，自动去掉表头
data = data.values


name = data[:,0]
features = data[:, 1:1001]  # 特征
labels = data[0:208,-1]   # 标签


# 分布归一化
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler = scaler.fit(features)  # 本质生成 max(x) 和 min(x)
features2 = scaler.transform(features)

features1 = features2[0:208,:]
features_valid1 =  features2[208:,:]

x_train,x_test,y_train,y_test = train_test_split(features1,labels,random_state=47,test_size=0.1)


# 标准化
from sklearn.preprocessing import StandardScaler
ss_x,ss_y = StandardScaler(),StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)
y_train = ss_y.fit_transform(y_train.reshape([-1,1])).reshape(-1)
y_test = ss_y.transform(y_test.reshape([-1,1])).reshape(-1)



# 线性回归
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
rf = clf.fit (x_train, y_train)
y_pred = rf.predict(x_test)
print("线性回归结果如下：")
print("训练集分数：",rf.score(x_train,y_train))
print("验证集分数：",rf.score(x_test,y_test))
print("*" * 100)


# 支持向量机回归
from sklearn.svm import SVR
#线性核函数
l_svr = SVR(kernel='linear',C=0.1)
rf = l_svr.fit(x_train,y_train)
# R2_1 = l_svr.score(x_test,y_test)
print("支持向量机回归（线性核函数）结果如下: ")
print("训练集分数：",rf.score(x_train,y_train))
print("验证集分数：",rf.score(x_test,y_test))
print("*" * 100)


# 多项式核函数
n_svr = SVR(kernel="poly",C=0.1)
rf = n_svr.fit(x_train,y_train)
# R2_2 = n_svr.score(x_test,y_test)
print("支持向量机回归（多项式核函数）结果如下：")
print("训练集分数：",rf.score(x_train,y_train))
print("验证集分数：",rf.score(x_test,y_test))
print("*" * 100)


# 径向基核函数
r_svr = SVR(kernel="rbf",C=0.1)
rf = r_svr.fit(x_train,y_train)
# R2_3 = r_svr.score(x_test,y_test)
print("支持向量机回归（径向基核函数）结果如下：")
print("训练集分数：",rf.score(x_train,y_train))
print("验证集分数：",rf.score(x_test,y_test))
print("*" * 100)


# K临近回归器
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(weights="uniform")
rf = knn.fit(x_train,y_train)
# R2_4 = knn.score(x_test,y_test)
print("K临近回归器结果如下：")
print("训练集分数：",rf.score(x_train,y_train))
print("验证集分数：",rf.score(x_test,y_test))
print("*" * 100)


# 回归树（决策树）
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
rf = dt.fit(x_train,y_train)
# R2_5 = dt.score(x_test,y_test)
print("回归树结果如下：")
print("训练集分数：",rf.score(x_train,y_train))
print("验证集分数：",rf.score(x_test,y_test))
print("*" * 100)


# 集成模型（3个）
# 随机森林
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rf = rfr.fit(x_train,y_train)
# R2_6 = rfr.score(x_test,y_test)
print("随机森林结果如下：")
print("训练集分数：",rf.score(x_train,y_train))
print("验证集分数：",rf.score(x_test,y_test))
# print('预测结果：',rf.predict(features_valid))
print("*" * 100)


# 极端森林
from sklearn.ensemble import ExtraTreesRegressor
etr = ExtraTreesRegressor()
rf = etr.fit(x_train,y_train)
# R2_7 = etr.score(x_test,y_test)
print("极端森林结果如下：")
print("训练集分数：",rf.score(x_train,y_train))
print("验证集分数：",rf.score(x_test,y_test))
# print('预测结果：',rf.predict(features_valid))
print("*" * 100)


# 提升树
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
rf = gbr.fit(x_train,y_train)
# R2_8 = gbr.score(x_test,y_test)
print("提升树结果如下：")
print("训练集分数：",rf.score(x_train,y_train))
print("验证集分数：",rf.score(x_test,y_test))
# print('预测结果：',rf.predict(features_valid))
print("*" * 100)



# 贝叶斯岭回归
from sklearn.linear_model import BayesianRidge, ARDRegression
clf = BayesianRidge()
rf = clf.fit(x_train,y_train)
# R2_9 = clf.score(x_test,y_test)
print("贝叶斯岭回归结果如下：")
print("训练集分数：",rf.score(x_train,y_train))  # 决定系数R^2
print("验证集分数：",rf.score(x_test,y_test))
print("*" * 100)


# 稀疏贝叶斯学习/相关向量机/ARD
ard = ARDRegression()
rf = rf = ard.fit(x_train,y_train)
# R2_10 = ard.score(x_test,y_test)
print("稀疏贝叶斯结果如下：")
print("训练集分数：",rf.score(x_train,y_train))
print("验证集分数：",rf.score(x_test,y_test))
print("*" * 100)


# BP神经网络
from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(500,), activation='logistic',random_state=31,learning_rate_init=0.1,max_iter=1000)  # BP神经网络回归模型
rf = model.fit(x_train,y_train)  # 训练模型
# R2_11 = model.score(x_test,y_test)
print("BP神经网络结果如下：")
print("训练集分数：",rf.score(x_train,y_train))
# print("验证集分数：",rf.score(x_test,y_test))
print('预测结果：',rf.predict(features_valid1))
print("*" * 100)


# LASSO回归模型(套索回归)
from sklearn.linear_model import Lasso
lasso = Lasso()
rf = lasso.fit (x_train, y_train)
# R2_12 = lasso.score(x_test,y_test)
print("LASSO回归模型结果如下：")
print("训练集分数：",rf.score(x_train,y_train))
print("验证集分数：",rf.score(x_test,y_test))
print("*" * 100)



# Bagging回归模型
from sklearn.ensemble import BaggingRegressor
clf = BaggingRegressor()
rf = clf.fit (x_train, y_train)
# y_pred = rf.predict(x_test)
print("Bagging回归模型结果如下：")
print("训练集分数：",rf.score(x_train,y_train))
print("验证集分数：",rf.score(x_test,y_test))
print("*" * 100)


# 梯度提升回归模型
from sklearn.ensemble import GradientBoostingRegressor
clf = GradientBoostingRegressor()
rf = clf.fit (x_train, y_train)
# y_pred = rf.predict(x_test)
print("梯度提升回归模型结果如下：")
print("训练集分数：",rf.score(x_train,y_train))
print("验证集分数：",rf.score(x_test,y_test))
print('预测结果：',rf.predict(features_valid))
print("*" * 100)


# AdaBoost回归模型
from sklearn.ensemble import AdaBoostRegressor
clf = AdaBoostRegressor()
rf = clf.fit (x_train, y_train)
# y_pred = rf.predict(x_test)
print("AdaBoost回归模型结果如下：")
print("训练集分数：",rf.score(x_train,y_train))
print("验证集分数：",rf.score(x_test,y_test))
print("*" * 100)


# 弹性网络回归(Elastic Net)
from sklearn.linear_model import ElasticNet
regr = ElasticNet(random_state=0,)
rf = regr.fit (x_train, y_train)
# y_pred = rf.predict(x_test)
print("弹性网络回归模型结果如下：")
print("训练集分数：",rf.score(x_train,y_train))
print("验证集分数：",rf.score(x_test,y_test))
print("*" * 100)


# 多层感知机回归模型
from sklearn.neural_network import MLPRegressor
clf = MLPRegressor()
rf = clf.fit (x_train, y_train)
# y_pred = rf.predict(x_test)
print("多层感知机回归模型结果如下：")
print("训练集分数：",rf.score(x_train,y_train))
print("验证集分数：",rf.score(x_test,y_test))
print("*" * 100)

# GBRT回归
from sklearn.ensemble import GradientBoostingRegressor
gr = GradientBoostingRegressor()
rf = gr.fit (x_train, y_train)
# y_pred = rf.predict(x_test)
print("GBRT回归结果如下：")
print("训练集分数：",rf.score(x_train,y_train))
print("验证集分数：",rf.score(x_test,y_test))
print('预测结果：',rf.predict(features_valid))
print("*" * 100)

