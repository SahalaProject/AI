# coding:utf-8

# 用训练好的模型来预测新的样本
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

"""
加载训练好的模型，识别本地图片--输出类别和相似度值
通过本地图片验证模型
"""

img_height = 28
img_width = 28
class_names =  ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] # 训练时类型

model = keras.models.load_model('./save_model/fashion_mnist') # 加载训练好的完整模型

# sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg" # 网络图片
# sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
sunflower_path = './image/Figure_1.png' # 磁盘图片路径

img = keras.preprocessing.image.load_img(sunflower_path, target_size=(img_height, img_width)) # 将图片加载为PIL格式
img_array = keras.preprocessing.image.img_to_array(img) # 将PIL映像实例转换为Numpy数组
img_array = tf.expand_dims(img_array, 0) # Create a batch # 使用expand_dims来将维度加1
print('img_array: ',img_array.shape)

predictions = model.predict(img_array) # 输入测试数据,输出预测结果
score = tf.nn.softmax(predictions[0])

class_index = int(np.argmax(score)) # 返回识别后最大值索引
class_score = 100 * np.max(score) # 相似度 返回数组的最大值或沿轴的最大值。
print("This image most likely belongs to {} with a {:.2f} percent confidence." .format(class_names[class_index], class_score))

