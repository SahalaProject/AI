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
fine_tune 迁移学习
resnet50 处理的是224*224的图片
weights='imagenet' # 有两个值，None从头训练, imagenet下载已经训练好的模型，通过这个训练好的模型的参数去初始化这个网络结构 
下载的model 的位置 'C:/Users\liyubin1\.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

"""

train_dir = './flower_5/training'
valid_dir = './flower_5/validation'
lable_file = './flower/lable.txt'
print(os.path.exists(train_dir))

# 为加载程序定义一些参数
batch_size = 24  # 图像变大调小
image_height = 224
image_width = 224
channels = 3  # 通道
num_classes = 5

# 读取数据
# lables = pd.read_csv(lable_file, header=0)
# 定义Generator
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=keras.applications.resnet50.preprocess_input, # 专门为resnet50设计的图像处理函数带有归一化，只在keras中，不在tf.keras中
                                                             rotation_range=40, # 将图像随机旋转
                                                             width_shift_range=0.2, # 水平方向 处理图像位移，在0-20%随机选择数
                                                             height_shift_range=0.2, # 它们0-1之间的数是比例， 大于1是像素值
                                                             shear_range=0.2,  # 剪切强度
                                                             zoom_range=0.2, # 缩放强度
                                                             horizontal_flip=True, # 是否做随机水平翻转
                                                             fill_mode='nearest' # 作用是当对图片处理放大时通过临近像素点near来填充真实的像素点
                                                             )
# 读取图片，通过上面定义的Generator对图片处理
train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                                    target_size=(image_height, image_width),
                                                    batch_size=batch_size, # 生成的图片以多少张为一组
                                                    seed=7, # 随机数
                                                    shuffle=True, # 数据是否需要滚爬
                                                    class_mode='categorical' # 用来控制lable的格式, categorical是one_hot编码后的lable
                                                    )

valid_datagen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=keras.applications.resnet50.preprocess_input) # 因为只是做验证。所以只需要做验证集的值的缩放，保证和训练集值分布一致
valid_generator = valid_datagen.flow_from_directory(directory=valid_dir,
                                                    target_size=(image_height, image_width),
                                                    batch_size=batch_size,
                                                    seed=7,
                                                    shuffle=False, # 唯一与训练集不同处，不需要训练所以不用做滚爬
                                                    class_mode='categorical'
                                                    )

# 查看训练集和验证集数据数量
train_num = train_generator.samples
valid_num = valid_generator.samples
print(train_num, valid_num)
# for i in range(3):
#     x, y = train_generator.next()
#     print(x.shape, y.shape)
#     print(y)
# (64, 128, 128, 3) (64, 5) # 如果未显示5分类，可能路径下还有一层文件夹
#  找到属于5个类别的3670个文件。 使用2936个文件进行训练。
# 在开发模型时，最好使用验证拆分。我们将使用80％的图像进行训练，并使用20％的图像进行验证。

#4、############################# resnet50_fine_tune 网络结构 模型训练######################################
resnet50_fine_tune = keras.models.Sequential()
resnet50_fine_tune.add(keras.applications.ResNet50(include_top=False, # 本身是处理千类问题的这里是10类，去掉最后一层
                                                   pooling='avg',
                                                   weights='imagenet' # 有两个值，None从头训练, imagenet下载已经训练好的模型，通过这个训练好的模型的参数去初始化这个网络结构
                                                   ))
resnet50_fine_tune.add(keras.layers.Dense(num_classes, activation='softmax'))
resnet50_fine_tune.layers[0].trainable = False

# 编译模型  计算目标函数 # 高级优化器adam快速稳定, 对于fine_tune来说sgd是个好的选择
# model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
# model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.SGD(0.001), metrics=['accuracy']) # optimizer模型的求解方法， metrics指标
resnet50_fine_tune.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) # optimizer模型的求解方法， metrics指标

resnet50_fine_tune.summary() # 模型结构

epochs = 20 # fine_tune 10次就可以达到比较好的效果
# 因为数据是 generator 产生出来的，所以调用fit_generator
history = resnet50_fine_tune.fit_generator(generator=train_generator,  # generator中传入产生train的generator
                              steps_per_epoch=train_num // batch_size, # 因为generator是不停的产生数据的，不知道是每个epoch有多少步step,所以需要显式的指出来
                              epochs=epochs,
                              validation_data=valid_generator,
                              validation_steps=valid_num // batch_size
                              )


resnet50_fine_tune.save(filepath='./save_model/flowers')
# 通过一张图打印出 训练值的变化过程
def polt_learning_curves(history):
    import pandas as pd
    pd.DataFrame(history.history).plot(figsize=(8, 5)) # DataFrame是pd中重要的数据结构, 图大小8和5
    plt.grid(True) # 显示网格
    plt.gca().set_ylim(0, 1)# 坐标轴范围
    plt.show()

polt_learning_curves(history) # 打印值训练值变化图
