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
 wide & deep模型 的 多输入 

keras子类API
功能API（函数式API）
多输入与多输出

回归模型  适合wide & deep模型，有8个特征可以划分给 wide 和 deep， 而图像分类中分类价值都一样，在次没意义
房价预测 数据集 （加尼福利亚的房价的数据集）
下载数据集失败，浏览器下载：https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/5976036/cal_housing.tgz
注释源码：urlretrieve(remote.url, file_path) 加载完成本地数据集后 取消注释

让一个分类模型变成回归模型： 
            keras.layers.Dense(1), # 输出一个数
            model.compile(loss='mean_squared_error'
"""

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
x_train_scaler = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

#############################wide&deep模型的多输入###########################################
# 搭建模型
# 多输入
input_wide = keras.layers.Input(shape=[5])
input_deep = keras.layers.Input(shape=[6])
hidden1 = keras.layers.Dense(30, activation='relu')(input_deep)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
concat = keras.layers.concatenate([input_wide, hidden2]) # 拼接起来
output = keras.layers.Dense(1)(concat) # 函数式调用
model = keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
# fit前需要拆分模型


print(model.summary())
# 编译模型  计算目标函数
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(0.001)) # 编译model
# 会调
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)]

# 因为有两个输入数据，需要拆分
x_train_scaled_wide= x_train_scaler[:, :5] # 取前5个
x_train_scaled_deep= x_train_scaler[:, 2:]
x_valid_scaled_wide = x_valid_scaled[:, :5]
x_valid_scaled_deep = x_valid_scaled[:, 2:]
x_test_scaled_wide = x_test_scaled[:, :5]
x_test_scaled_deep = x_test_scaled[:, 2:]


# 训练, 多输入x替换
history = model.fit([x_train_scaled_wide, x_train_scaled_deep], y_train,
                    validation_data=([x_valid_scaled_wide, x_valid_scaled_deep], y_valid),
                    epochs=100, callbacks=callbacks)

# 通过一张图打印出 训练值的变化过程
def polt_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5)) # DataFrame是pd中重要的数据结构, 图大小8和5
    plt.grid(True) # 显示网格
    plt.gca().set_ylim(0, 1)# 坐标轴范围, 如果图显示不全调整下x,y轴
    plt.show()

polt_learning_curves(history) # 打印值训练值变化图

# 测试模型
test_result = model.evaluate([x_test_scaled_wide, x_test_scaled_deep], y_test)
print(test_result)