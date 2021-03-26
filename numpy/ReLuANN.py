# encoding: utf-8

"""测试结果
sigmoid：
time: 107.68768955199994 s
0 : 0.9948979591836735
1 : 0.9903083700440528
2 : 0.9699612403100775
3 : 0.9752475247524752
4 : 0.9735234215885947
5 : 0.976457399103139
6 : 0.9697286012526096
7 : 0.9659533073929961
8 : 0.9661190965092402
9 : 0.9722497522299306
average: 0.9756
ReLu

"""

import numpy as np
import scipy.special
import scipy.ndimage
import matplotlib.pyplot as plt
import time
from progressbar import *

start = time.perf_counter()


# 一些激活函数
# 阶跃函数
def step_function(x):
    y = x > 0
    return y.astype(np.int)


# sigmoid常用于二元分类的输出函数
def sigmoid(x):
    return scipy.special.expit(x)
'''sigmoid表达式
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
'''


def diff_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


def inverse_sigmoid(y):
    return scipy.special.logit(y)
'''
def inverse_sigmoid(y):
    y = y / (1 - y)
    return np.log(y)
'''


# 现阶段流行的激活函数
def ReLu(x):
    return np.maximum(x, 0.0)


def diff_ReLu(w):
    return step_function(w)


# 常用于多元分类的输出函数，输出和为1，所以可以称为置信概率，可以省略
def softmax(x):
    c = np.max(x)
    y = np.exp(x - c) / np.sum(np.exp(x - c))
    return y


# 恒等函数，常用于回归问题中
def identity_func(x):
    return x


# 读取mnist数据集的函数
def load_mnist(data_file):
    # 读取文件数据
    train_data_file = open(data_file + "mnist_train.csv", 'r')
    train_data_list = train_data_file.readlines()
    train_data_file.close()

    test_data_file = open(data_file + "mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # 数据切片，分类图片和标签
    train_imgs = []
    train_labels = []
    for record in train_data_list:
        all_values = record.split(',')
        train_imgs.append(all_values[1:])
        train_labels.append(all_values[0])
        pass
    train_imgs = np.asfarray(train_imgs)
    train_labels = np.asfarray(train_labels)

    test_imgs = []
    test_labels = []
    for record in test_data_list:
        all_values = record.split(',')
        test_imgs.append(all_values[1:])
        test_labels.append(all_values[0])
        pass
    test_imgs = np.asfarray(test_imgs)
    test_labels = np.asfarray(test_labels)

    return (train_imgs, train_labels), (test_imgs, test_labels)


# 神经网络模型定义

class ANN:

    # 初始化
    def __init__(self, layer0_nodes, layer1_nodes, layer2_nodes,  learning_rate):
        # 接收参数
        self.nodes_0 = layer0_nodes
        self.nodes_1 = layer1_nodes
        self.nodes_2 = layer2_nodes
        # self.nodes_3 = layer3_nodes
        self.lr = learning_rate

        # 使用正态分布随即初始化权重,中心值为0，标准方差为后一层节点数开平方的倒数
        # 权重矩阵行表示连接后一节点的所有权重，列表示连接前一节点的所有权重
        self.w01 = np.random.normal(0.0, pow(self.nodes_1, -0.5), (self.nodes_1, self.nodes_0))
        self.w12 = np.random.normal(0.0, pow(self.nodes_2, -0.5), (self.nodes_2, self.nodes_1))
        # self.w23 =
        # 设置激活函数
        self.activation_1 = lambda x: ReLu(x)
        self.activation_2 = lambda x: ReLu(x)
        self.diff_1 = lambda x: diff_ReLu(x)
        self.diff_2 = lambda x: diff_ReLu(x)
        self.inverse_activation_1 = lambda x: inverse_sigmoid(x)
        self.inverse_activation_2 = lambda x: inverse_sigmoid(x)
        pass

    # 训练模型
    def train(self, img, label):
        # 转置
        inputs_0 = np.array(img, ndmin=2).T

        # 将label转化为目标矩阵
        targets = np.zeros(self.nodes_2)
        targets[label] = 1
        targets = np.array(targets, ndmin=2).T

        # 前馈数据，计算结果
        inputs_1 = np.dot(self.w01, inputs_0)
        outputs_1 = self.activation_1(inputs_1)
        inputs_2 = np.dot(self.w12, outputs_1)
        outputs_2 = self.activation_2(inputs_2)

        # 误差
        erro_2 = targets - outputs_2
        erro_1 = np.dot(self.w12.T, erro_2)

        # 反馈调参
        self.w12 += self.lr * np.dot(erro_2 * self.diff_2(inputs_2), outputs_1.T)
        self.w01 += self.lr * np.dot(erro_1 * self.diff_1(inputs_1), inputs_0.T)
        pass

    # 查询
    def query(self, img):
        # 转置
        inputs_0 = np.array(img, ndmin=2).T

        # 前馈数据，计算结果,结果为概率
        inputs_1 = np.dot(self.w01, inputs_0)
        outputs_1 = self.activation_1(inputs_1)
        inputs_2 = np.dot(self.w12, outputs_1)
        outputs_2 = self.activation_2(inputs_2)
        return outputs_2

    # 逆向查询，分析模型
    def back_query(self, label):
        # 将label转化为目标矩阵
        targets = np.zeros(self.nodes_2)
        targets[label] = 1
        outputs_2 = np.array(targets, ndmin=2).T

        # 逆向查询，计算结果
        inputs_2 = self.inverse_activation_2(outputs_2)
        outputs_1 = np.dot(self.w12.T, inputs_2)
        inputs_1 = self.inverse_activation_1(outputs_1)
        inputs_0 = np.dot(self.w01.T, inputs_1)
        return inputs_0
    pass


# 主程序

# 构建模型
nodes_0 = 28*28
nodes_1 = 200
nodes_2 = 10
learning_rate = 0.1
epochs = 5

ann = ANN(nodes_0, nodes_1, nodes_2, learning_rate)

# 导入数据集
(train_imgs, train_labels), (test_imgs, test_labels) = load_mnist("/home/nieyan/notebook/datasets/mnist/")

# 数据预处理
(train_imgs, test_imgs) = (train_imgs / 255.0, test_imgs / 255.0)


# 训练
start = time.perf_counter()

for i in range(epochs):
    print(i + 1)
    pbar = ProgressBar(maxval=10 * train_labels.size + 1).start()
    for index in range(train_labels.size):
        pbar.update(10 * index + 1)
        ann.train(train_imgs[index], int(train_labels[index]))
        pass
    pbar.finish()
    pass

end = time.perf_counter()
print("time:", end - start, "s")


# 测试
scorecard = [[], [], [], [], [], [], [], [], [], []]
for index in range(test_labels.size):
    outputs = ann.query(test_imgs[index])
    label = np.argmax(outputs)
    correct_label = int(test_labels[index])
    if label == correct_label:
        scorecard[correct_label].append(1)
    else:
        scorecard[correct_label].append(0)
        pass
    pass

for label in range(10):
    # 计算得分
    print(label, ":", scorecard[label].count(1) / len(scorecard[label]))
    pass

# 给列表数据降维
list = []
for a in scorecard:
    for b in a:
        list.append(b)
        pass
    pass

scorecard_array = np.array(list)

print("average:", scorecard_array.sum() / scorecard_array.size)

# 逆向查询，从最后层输入数据，逆向计算出输入的图片，就可以直观查看模型从图片中学习到的分类特征

# 构建收集图像数据的数组
image_data = np.zeros(shape=(10, 28, 28), dtype=np.float64)

for label in range(10):
    # 执行逆向查询，并将收到的数据转化为图像矩阵的格式
    image_data[label] = ann.back_query(label).reshape(28, 28)
    pass

# 绘制输出图像
plt.figure(figsize=(10, 5))
for a in range(10):
    plt.subplot(2, 5, a+1)
    plt.imshow(image_data[a], cmap='Greys', interpolation='None')
    plt.xlabel(a)
plt.show()
