# coding:utf-8

from tensorflow import keras

"""从保存的模型重新加载新的 Keras 模型"""


new_model = keras.models.load_model('./save_model/mnist')
print(new_model.summary()) # # 检查其架构


# 还原的模型使用与原始模型相同的参数进行编译。 尝试使用加载的模型运行评估和预测：
mnist = keras.datasets.mnist # 导出数据集合
(x_train_all, y_train_all), (x_test, y_test) = mnist.load_data() # 拆分数据集 和 测试集, 下载会报错，看源码浏览器下载包，放入加载地址/Users/li/.keras/datasets/fashion-mnist

loss, acc = new_model.evaluate(x_test, y_test)  # 验证准确率

print('Restored model, accuracy: {:5.2f}%'.format(100*acc))
print(new_model.predict(x_test).shape)
