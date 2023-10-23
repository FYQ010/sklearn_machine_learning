# 自整理使用的sklearn实现的机器学习分类算法
# 以下代码是针对我的数据进行编写，如需使用请替换数据

import seaborn as sns; sns.set()
import time

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier



file_path = r'D:\Phyon\Project\bys/label_data_change.xlsx'  # 读取表格
data = pd.read_excel(file_path,header=0)  # header为表头，自动去掉表头
data = data.values

features = data[:, 1:9]  # 特征
labels = data[:, -3]

features_train, features_test = train_test_split(features, test_size=0.2, random_state=42)
labels_train, labels_test = train_test_split(labels, test_size=0.2, random_state=42)


# 高斯朴素贝叶斯模型
from sklearn.naive_bayes import GaussianNB
print('开始训练高斯朴素贝叶斯 | ','时间：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
clf1 = GaussianNB()  #高斯朴素贝叶斯模型
clf1.fit(features_train, labels_train)
print("训练完毕 | 结果如下: ")
print("训练集分数：",clf1.score(features_train,labels_train))
print("验证集分数：",clf1.score(features_test,labels_test))
print("--" * 100)

labels_test = clf1.predict(features_test)

# 伯努利朴素贝叶斯
from sklearn.naive_bayes import BernoulliNB
print('开始训练伯努利朴素贝叶斯 | ','时间：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
clf2 = BernoulliNB() # 伯努利朴素贝叶斯
clf2.fit(features_train, labels_train)
print("训练完毕 | 结果如下: ")
print("训练集分数：",clf2.score(features_train,labels_train))
print("验证集分数：",clf2.score(features_test,labels_test))
print("--" * 100)



# 多项式朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB
print('开始训练多项式朴素贝叶斯 | ','时间：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
clf3 = MultinomialNB() # 多项式朴素贝叶斯
clf3.fit(features_train, labels_train)
print("训练完毕 | 结果如下: ")
print("训练集分数：",clf3.score(features_train,labels_train))
print("验证集分数：",clf3.score(features_test,labels_test))
print("--" * 100)



# 支持向量机
from sklearn.svm import SVC
print('开始训练支持向量机 | ','时间：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
clf4 = SVC()
clf4.fit(features_train, labels_train)
print("训练完毕 | 结果如下: ")
print("训练集分数：",clf4.score(features_train,labels_train))
print("验证集分数：",clf4.score(features_test,labels_test))
print("--" * 100)


# 支持向量机(linear)
from sklearn.svm import LinearSVC
print('开始训练支持向量机(linear) | ','时间：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
clf5 = LinearSVC()
clf5.fit(features_train, labels_train)
print("训练完毕 | 结果如下: ")
print("训练集分数：",clf5.score(features_train,labels_train))
print("验证集分数：",clf5.score(features_test,labels_test))
print("--" * 100)


# 支持向量机(linear)
from sklearn.svm import LinearSVC
print('开始训练支持向量机 | ','时间：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
clf5 = LinearSVC()
clf5.fit(features_train, labels_train)
print("训练完毕 | 结果如下: ")
print("训练集分数：",clf5.score(features_train,labels_train))
print("验证集分数：",clf5.score(features_test,labels_test))
print("--" * 100)



# 随机森林
from sklearn.ensemble import RandomForestClassifier
print('开始训练随机森林 | ','时间：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
clf6 = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
clf6.fit(features_train, labels_train)
# result = clf6.predict(features_test)
# print(result)
print("训练完毕 | 结果如下: ")
print("训练集分数：",clf6.score(features_train,labels_train))
print("验证集分数：",clf6.score(features_test,labels_test))
print("--" * 100)



# 决策树
from sklearn import tree
print('开始训练决策树 | ','时间：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
clf7 = tree.DecisionTreeClassifier()
clf7.fit(features_train, labels_train)
print("训练完毕 | 结果如下: ")
print("训练集分数：",clf7.score(features_train,labels_train))
print("验证集分数：",clf7.score(features_test,labels_test))
print("--" * 100)



# KNN分类
from sklearn.neighbors import KNeighborsClassifier
print('开始训练KNN | ','时间：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
clf8 = KNeighborsClassifier()
clf8.fit(features_train, labels_train)
print("训练完毕 | 结果如下: ")
print("训练集分数：",clf8.score(features_train,labels_train))
print("验证集分数：",clf8.score(features_test,labels_test))
print("--" * 100)



# 神经网络
from sklearn.neural_network import MLPClassifier
print('开始训练神经网络 | ','时间：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
clf9 = MLPClassifier(max_iter=1000, learning_rate_init=0.002)
clf9.fit(features_train, labels_train)
print("训练完毕 | 结果如下: ")
print("训练集分数：",clf9.score(features_train,labels_train))
print("验证集分数：",clf9.score(features_test,labels_test))
print("--" * 100)


