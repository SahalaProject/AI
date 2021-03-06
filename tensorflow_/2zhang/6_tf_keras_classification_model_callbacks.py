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
keras 回调函数
Tensorboard  (需要文件夹)
EarlyStopping  及时停止不达目标的训练
ModelCheckpoint （需要文件名）
在运行过程中监听/做其他动作
详见：官方api
打开TensorBoard查看，callbacks/下 命令行： tensorboard --logdir=callbacks (pwd的路径，相对路径无效) , 将会构建个人服务器  http://localhost:6006/ 
"""

#1、#############################分类模型数之据读取与展示######################################
# 分类数据集 fashion_mnist 60000张图片
fashion_mnist = keras.datasets.fashion_mnist # 导出数据集合
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data() # 拆分数据集 和 测试集, 下载会报错，看源码浏览器下载包，放入加载地址/Users/li/.keras/datasets/fashion-mnist

# 训练集拆分：训练集 和 验证集
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]  # x 前55000张作为验证集，后5000张作为训练集
y_valid, y_train = y_train_all[:5000], y_train_all[5000:] #

# 打印训练集/验证集/测试集 的 shape它们都是numpy格式
print(x_train.shape, y_train.shape) # (55000, 28, 28) (55000,)
print(x_valid.shape, y_valid.shape) # (5000, 28, 28) (5000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

# 归一化前  打印训练集中最大值最小值
print(np.max(x_train), np.min(x_train)) # 255 0
#4、#############################分类模型之数据归一化######################################
# 对数据做归一化, 验证集/测试集做归一化时也需要用训练集的均值和方差做，这样才能达到一个好的效果
# x = (x - u) / std # u是均值， std是方差， 结果就是一个均值是1，方差是0 的正态分布了
from sklearn.preprocessing import StandardScaler # 计算训练集的平均值和标准差,以便测试数据集使用相同的变换
scaler = StandardScaler() # 初始化
#给训练集做归一化 使用fit_transform ， x_train: [None, 28, 28] 三维矩阵转 2维(reshape(-1, 1)) --> [None, 784] ， 最后再转回来 reshape(-1, 28, 28)
x_train_scaled = scaler.fit_transform(x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)# 对训练集做归一化, 因为是int 所以转np.float32
# 给验证集做归一化
x_valid_scaled = scaler.transform(x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
# 给测试集做归一化, 使用和训练集一样的 均值 和 方差
x_test_scaled = scaler.transform(x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
# 这样就将训练集/验证集/测试集做成了归一化后的
# 打印归一化后， 训练集最大值和最小值
print(np.max(x_train_scaled), np.min(x_train_scaled)) # 2.0231433 -0.8105136


#2、#############################分类模型之模型构建######################################
"""
model = keras.models.Sequential() # 创建Sequential对象
# 第一层
model.add(keras.layers.Flatten(input_shape=[28, 28])) # 添加层，首先输入层 通过Flatten将输入的图片，将28x28的二维矩阵 展平 成 28x28的 一维向量
# 全连接层： 就是神经网络中最普通的一种神经网络，通过层次放置神经网络，下一层的所有单元 和上一层的所有单元都一一的进行连接
model.add(keras.layers.Dense(units=300, activation='relu')) # 加入全连接层 单元数300, activation计划函数
model.add(keras.layers.Dense(units=100, activation='relu')) # 100个单元去和300个全连接
model.add(keras.layers.Dense(10, activation='softmax')) # 输出长度为10的向量
# relu: y = max(0, x) # 取最大值
# softmax: 将向量变成概率分布。 x = [x1, x2, x3]
#          y = [e^x1/sum, e^xx2/sum, e^x3/sum], sum = e^x1 + e^x2 + e^x3
"""
# 归一化后， 重新构建模型
model = keras.models.Sequential([keras.layers.Flatten(input_shape=[28, 28]),
                                 keras.layers.Dense(units=300, activation='relu'),
                                 keras.layers.Dense(units=100, activation='relu'),
                                 keras.layers.Dense(10, activation='softmax')
                                 ]) # 创建Sequential对象 时将值传入

# 编译模型  计算目标函数
# 如何选loss：reason for sparse: y-->index , 算子:  y-> one_hot ->[向量] ， 如果y已经是向量那只用 categorical_crossentropy， 反之在这里y只是个数用sparse_categorical_crossentropy
# 调用这个model.compile函数的目的是： 将 损失函数/优化方法/metrics 加到图中去，同时将图固化下来
model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.SGD(0.001), metrics=['accuracy']) # optimizer模型的求解方法， metrics指标

# print(model.layers)  # 查看模型有多少层, 目前四层

# model.summary()  # 查看模型概况， 模型架构图 四层

# 235500 如何得来 [None, 784] * W + b -> [None, 300] W.shape [784, 300], b = [300]
# 235500 = 784 x 300 + 300
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# flatten (Flatten)            (None, 784)               0
# _________________________________________________________________
# dense (Dense)                (None, 300)               235500
# _________________________________________________________________
# dense_1 (Dense)              (None, 100)               30100
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                1010
# =================================================================
# Total params: 266,610
# Trainable params: 266,610
# Non-trainable params: 0

############################## callbacks回调函数监听 ######################################
# 在fit前调用 ， Tensorboard ,  EarlyStopping , ModelCheckpoint
logdir = './callbacks' # 定义文件夹
if not os.path.exists(logdir):
    os.mkdir(logdir)
# 定义输出的model文件
output_model_file = os.path.join(logdir, 'fashion_mnist_model.h5')

callbacks = [
    keras.callbacks.TensorBoard(logdir), # 定义callbacks
    keras.callbacks.ModelCheckpoint(output_model_file, save_best_only=True),  # save_best_only保存最佳模型，默认最近一个
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
]
# 运行结束后 查看 callbacks 的信息
# 打开TensorBoard查看： tensorboard --logdir=callbacks

#3、#############################模型训练######################################
# history ， fit返回中间运行的结果,  之所以这么称呼是因为该方法使模型“适合”训练数据：
history = model.fit(x_train_scaled, y_train, epochs=100, validation_data=(x_valid_scaled, y_valid), callbacks=callbacks) # epochs遍历数据集的次数, 每隔一段时间将会对验证集做验证

# print(history.history)
# print(type(history)) # callbacks

# 测试模型
test_result = model.evaluate(x_test_scaled, y_test)
print(test_result) # loss损失 和 准确率  [0.42931729555130005, 0.84579998254776]

# 通过一张图打印出 训练值的变化过程
def polt_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5)) # DataFrame是pd中重要的数据结构, 图大小8和5
    plt.grid(True) # 显示网格
    plt.gca().set_ylim(0, 3)# 坐标轴范围, 如果图显示不全调整下x,y轴
    plt.show()

polt_learning_curves(history) # 打印值训练值变化图


