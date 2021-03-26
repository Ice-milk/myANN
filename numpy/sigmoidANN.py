# encoding: utf-8

'''测试结果
time: 464.7174529529984
0 : 0.9711155378486056
1 : 0.9894179894179894
2 : 0.9822485207100592
3 : 0.9545014520813165
4 : 0.9753340184994861
5 : 0.9781609195402299
6 : 0.9701030927835051
7 : 0.9774066797642437
8 : 0.963265306122449
9 : 0.9611553784860558
average: 0.9724
date: 2021.3.10
'''

import numpy as np
import scipy.special
import scipy.ndimage
import matplotlib.pyplot as plt
import time


# 神经网络类定义
class neuralNetwork:

    # 初始化
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 设置不同层的节点数量
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 设置学习率
        self.lr = learningrate

        # 初始化链接矩阵wih和who,正态分布方法，中心值为0，标准方差为后层节点数开平方的倒数，
        # 行矩阵表示传入后层节点的所有链接权重，列矩阵为前层节点所有的链接权重
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 设置激活函数sigmoid()
        self.activation_func = lambda x: scipy.special.expit(x)
        # 逆激活函数，用于逆向分析模型
        self.inverse_activation_func = lambda x: scipy.special.logit(x)
        pass

    # 训练
    def train(self, inputs_list, targets_list):
        # 输入矩阵化并转置,ndmin表示至少有2维，不足则自动增维，否则一维数组无法转置
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # 前馈网络
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        # 反馈调参
        # 误差
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        # 根据误差调整链接权重w
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), hidden_outputs.T)
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), inputs.T)
        pass

    # 查询
    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        # 前馈网络
        hidden_outputs = self.activation_func(np.dot(self.wih, inputs))
        final_outputs = self.activation_func(np.dot(self.who, hidden_outputs))
        return final_outputs

    # 逆向查询
    def backquery(self, targets_list):
        final_outputs = np.array(targets_list, ndmin=2).T
        # 使用逆向激活函数求输入
        final_inputs = self.inverse_activation_func(final_outputs)

        # 求隐藏层输出
        hidden_outputs = np.dot(self.who.T, final_inputs)
        # 使用逆向激活函数前需要数据规范化，数据大小在0.01-0.99之间
        hidden_outputs -= np.min(hidden_outputs)
        hidden_outputs = hidden_outputs / np.max(hidden_outputs) * 0.98 + 0.01
        hidden_inputs = self.inverse_activation_func(hidden_outputs)

        inputs = np.dot(self.wih.T, hidden_inputs)
        # 返回输入之前，将输入规范到0-1之间
        inputs -= np.min(inputs)
        inputs = inputs / np.max(inputs) * 255
        return inputs


# 定义不同层的节点数
input_nodes = 28*28
hidden_nodes = 200
output_nodes = 10

# 定义学习率
learning_rate = 0.1

# 训练循环次数
epochs = 5

# 创建一个神经网络
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 测试
# n.query([1.0, 0.5, 1-1.5])

# 导入数据集
# 读取csv文件
data_file = open("/home/nieyan/notebook/datasets/mnist/mnist_train.csv", 'r')
data_list = data_file.readlines()
data_file.close()

# 训练模型
start = time.perf_counter()

for e in range(epochs):
    for record in data_list:
        # 整理数据
        all_values = record.split(',')
        inputs = np.asfarray(all_values[1:]) / 255.0 * 0.98 + 0.01
        # 利用标签，构建目标矩阵
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        # 训练模型
        n.train(inputs, targets)

        # 构建旋转10度的图片，再次训练模型
        inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), 10, cval=0.01, order=1,
                                                              reshape=False)
        n.train(inputs_plusx_img.reshape(784), targets)
        inputs_minusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), -10, cval=0.01, order=1,
                                                               reshape=False)
        n.train(inputs_minusx_img.reshape(784), targets)
        pass
    pass

end = time.perf_counter()
print("time:", end - start)

# 测试模型
# 加载测试数据
test_data_file = open("/home/nieyan/notebook/datasets/mnist/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# 多次测试并记录得分衡量模型
# 初始化得分数组
scorecard = [[], [], [], [], [], [], [], [], [], []]

# 用整个测试集测试模型并记录得分
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    # print(correct_label, "correct label")
    inputs = np.asfarray(all_values[1:]) / 255.0 * 0.98 + 0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    # print(label, "network's answer")
    if label == correct_label:
        scorecard[label].append(1)
    else:
        scorecard[label].append(0)
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
    # 构建输入的目标数组
    targets = np.zeros(output_nodes) + 0.01
    targets[label] = 0.99
    # 执行逆向查询，并将收到的数据转化为图像矩阵的格式
    image_data[label] = n.backquery(targets).reshape(28, 28)
    pass

# 绘制输出图像
plt.figure(figsize=(10, 5))
for a in range(10):
    plt.subplot(2, 5, a+1)
    plt.imshow(image_data[a], cmap='Greys', interpolation='None')
    plt.xlabel(a)
plt.show()
