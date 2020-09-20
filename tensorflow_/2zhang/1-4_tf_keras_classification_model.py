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
分类模型创建/训练
数据集下载失败：浏览器下载， 然后注销源码下载步骤 paths.append(get_file(fname, origin=base + fname, cache_subdir=dirname)) ，
运行代码完成本地加载后，取消注释
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

# 得到数据集以后我们一般看一下里面图像是什么样的，有助于让我们了解数据集 也是机器学习很重要的一部分
def shou_single_image(img_arr):
    """
    展示数据集中图片
    :param img_arr: img是numpy数组
    :return:
    """
    plt.imshow(img_arr, cmap='binary') # cmap颜色图谱，默认是rgb, 这里是黑白我们用binary二位图显示就可以
    plt.show() # 展示图片

# shou_single_image(x_train[1]) # 展示训练集第一张图

# 更多图
def show_imgs(n_rows, n_cols, x_data, y_data, class_names):
    """

    :param n_rows:
    :param n_cols:
    :param x_data: 显示图这个参数就可以
    :param y_data: 图片对应类别名
    :param class_names: 类别对应的索引
    :return:
    """
    assert len(x_data) == len(y_data) # 验证样本数一致
    assert n_rows * n_cols < len(x_data) # 行X列 <= 样本数
    plt.figure(figsize=(n_cols * 1.4, n_rows * 1.6))  # 用plt定义一张大图
    # 对每一行每一列放一张图片
    for row in range(n_rows):
        for col in range(n_cols):
            index =  n_cols * row + col# 计算当前位置上放的图片的索引,(n_cols * row当前放慢的图片 + col就是这张图索引)
            plt.subplot(n_rows, n_cols, index+1) # 在这张大图上画上子图,(这里index是1开始，我们的index是从0开始所以+1)
            plt.imshow(x_data[index], cmap='binary', interpolation='nearest') # interpolation 缩放图片时差值方法， nearest最近的差值点作为我们像素点差值的值

            plt.axis('off') # 因为小图都放到了大图，所以不需要坐标系了
            plt.title(class_names[y_data[index]]) # 给小图们起个title

    plt.show() # 展示大图

# 定义class_names 衣服类型
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# 3行5列展示一张15张小图的大图, 还有对应的类别
# show_imgs(3, 3, x_train, y_train, class_names)


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
# 上面模型创建另一种写法
model = keras.models.Sequential([keras.layers.Flatten(input_shape=[28, 28]),
                                 keras.layers.Dense(units=300, activation='relu'),
                                 keras.layers.Dense(units=100, activation='relu'),
                                 keras.layers.Dense(10, activation='softmax')
                                 ]) # 创建Sequential对象 时将值传入

# 编译模型  计算目标函数
# 如何选loss：reason for sparse: y-->index , 算子:  y-> one_hot ->[向量] ， 如果y已经是向量那只用 categorical_crossentropy， 反之在这里y只是个数用sparse_categorical_crossentropy
# 调用这个model.compile函数的目的是： 将 损失函数/优化方法/metrics 加到图中去，同时将图固化下来
model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.SGD(0.001), metrics=['accuracy']) # optimizer模型的求解方法， metrics指标

print(model.layers)  # 查看模型有多少层, 目前四层

model.summary()  # 查看模型概况， 模型架构图 四层

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


#3、#############################模型训练######################################
# history ， fit返回中间运行的结果, 之所以这么称呼是因为该方法使模型“适合”训练数据：
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_valid, y_valid)) # epochs遍历数据集的次数, 每隔一段时间将会对验证集做验证

# print(history.history)
# print(type(history)) # callbacks

# 测试模型
model.evaluate(x_test, y_test)

# 通过一张图打印出 训练值的变化过程
def polt_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5)) # DataFrame是pd中重要的数据结构, 图大小8和5
    plt.grid(True) # 显示网格
    plt.gca().set_ylim(0, 1)# 坐标轴范围
    plt.show()

polt_learning_curves(history) # 打印值训练值变化图



