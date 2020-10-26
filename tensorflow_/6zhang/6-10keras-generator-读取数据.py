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


train_dir = './flower_5/training'
valid_dir = './flower_5/validation'
lable_file = './flower/lable.txt'
print(os.path.exists(train_dir))

# 为加载程序定义一些参数
batch_size = 32
image_height = 28
image_width = 28
channels = 3  # 通道
num_classes = 5

# 读取数据
# lables = pd.read_csv(lable_file, header=0)
# 定义Generator
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, # 图片中每一个像素点乘255
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

valid_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255) # 因为只是做验证。所以只需要做验证集的值的缩放，保证和训练集值分布一致
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

#4、#############################cnn卷积模型训练######################################
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(image_height, image_width, channels)))
model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.AlphaDropout(rate=0.5))
model.add(keras.layers.Dense(num_classes, activation='softmax'))


# 编译模型  计算目标函数 # 高级优化器adam快速稳定
# model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
# model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.SGD(0.001), metrics=['accuracy']) # optimizer模型的求解方法， metrics指标
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # optimizer模型的求解方法， metrics指标

model.summary() # 模型结构

epochs = 50
# 因为数据是 generator 产生出来的，所以调用fit_generator
history = model.fit_generator(generator=train_generator,  # generator中传入产生train的generator
                              steps_per_epoch=train_num // batch_size, # 因为generator是不停的产生数据的，不知道是每个epoch有多少步step,所以需要显式的指出来
                              epochs=epochs,
                              validation_data=valid_generator,
                              validation_steps=valid_num // batch_size
                              )


model.save(filepath='./save_model/flowers')
# 通过一张图打印出 训练值的变化过程
def polt_learning_curves(history):
    import pandas as pd
    pd.DataFrame(history.history).plot(figsize=(8, 5)) # DataFrame是pd中重要的数据结构, 图大小8和5
    plt.grid(True) # 显示网格
    plt.gca().set_ylim(0, 1)# 坐标轴范围
    plt.show()

polt_learning_curves(history) # 打印值训练值变化图
