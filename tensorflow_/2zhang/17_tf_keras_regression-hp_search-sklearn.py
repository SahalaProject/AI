# coding:utf-8

import matplotlib as mpl
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf

from tensorflow import keras

# 打印name和版本
print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)

"""
超参数搜索 使用sklearn实现

回归模型
房价预测 数据集 （加尼福利亚的房价的数据集）
下载数据集失败，浏览器下载：https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/5976036/cal_housing.tgz
注释源码：urlretrieve(remote.url, file_path) 加载完成本地数据集后 取消注释

让一个分类模型变成回归模型： 
            keras.layers.Dense(1), # 输出一个数
            model.compile(loss='mean_squared_error'
"""

# 通过一张图打印出 训练值的变化过程
def polt_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5)) # DataFrame是pd中重要的数据结构, 图大小8和5
    plt.grid(True) # 显示网格
    plt.gca().set_ylim(0, 1)# 坐标轴范围, 如果图显示不全调整下x,y轴
    plt.show()


from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing() # 获取数据
# 了解数据集
# print(housing.DESCR)
# print(housing.data.shape) # 数据大小 ，相当于 x
# print(housing.target.shape) # target大小 ，相当于 y

# 查看数据集 数据
import pprint # 打印结果会好看 pprint
# pprint.pprint(housing.data[0:5]) # x
# pprint.pprint(housing.target[0:5]) # y

# 划分样本
from sklearn.model_selection import train_test_split

# 拆分训练集 / 测试集 , 默认情况train_test_split将数据以3：1 的方式划分，（后面的是4：1），默认就是test_size=0.25
x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data, housing.target, random_state=7, test_size=0.25)
# 拆分训练集 / 验证集
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state=11)

print(x_train.shape, y_train.shape) # 查看样本
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)

# 归一化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

##################################sklearn实现超参搜索#################################################
# RandommizeSearchCV
# 1.转化为sklearn的model
# 2.定义参数集合
# 3.搜索参数

def build_model(hidden_layers=1, layear_size=30, learning_rate=3e-3):
    """定义model"""
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(30, activation='relu', input_shape=x_train.shape[1:]))  # 输出一个数

    for _ in range(hidden_layers - 1):
        model.add(keras.layers.Dense(layear_size, activation='relu'))
        model.add(keras.layers.Dense(1))
        optimizer = keras.optimizers.SGD(learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

# 转换sklearn的model
sklearn_model = keras.wrappers.scikit_learn.KerasRegressor(build_fn=build_model)
# 会调
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)]

history = sklearn_model.fit(x_train_scaled, y_train, epochs = 10, validation_data = (x_valid_scaled, y_valid), callbacks = callbacks)

###################################################################################################

polt_learning_curves(history) # 打印值训练值变化图

# 测试模型
test_result = sklearn_model.evaluate(x_test_scaled, y_test)
print(test_result)

