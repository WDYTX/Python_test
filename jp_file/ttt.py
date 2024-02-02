# import torch
# a = torch.ones(12).reshape(3, 4)
# b = torch.arange(12.).reshape(3, 4)
# print(a)
# a=torch.tensor([[1,2,3,4],[1,2,34,4],[1,23,4,5],[12,23,4,123],[1,2,34,4]])
# print(len(a))
# print(a.shape)
# # c=torch.cat([a,b],1)
# # print(c)
# for i in range(1,10,2):
#     print(i)
# %matplotlib inline
# import random
# import torch
# from d2l import torch as d2l
# def synthetic_data(w, b, num_examples): #@save
# #⽣成y=Xw+b+噪声"""
#     X = torch.normal(0, 1, (num_examples, len(w)))
#     y = torch.matmul(X, w) + b
#     y += torch.normal(0, 0.01, y.shape)
#     return X, y.reshape((-1, 1))
# true_w = torch.tensor([2, -3.4])
# true_b = 4.2
# features, labels = synthetic_data(true_w, true_b, 1000)
# print('features:', features[0],'\nlabel:', labels[0])
# d2l.set_figsize()
# d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
# def data_iter(batch_size, features, labels):
#     num_examples = len(features)
#     indices = list(range(num_examples))
#     # 这些样本是随机读取的，没有特定的顺序
#     random.shuffle(indices)
#     for i in range(0, num_examples, batch_size):
#         batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
#         yield features[batch_indices], labels[batch_indices]
# batch_size = 10
# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break
# w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
# print(w)
# b = torch.zeros(1, requires_grad=True)
# print(b)
# def linreg(X, w, b): #@save
#     #线性回归模型"""
#     return torch.matmul(X, w) + b
# def squared_loss(y_hat, y): #@save
#     """均⽅损失"""
#     return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
# def sgd(params, lr, batch_size): #@save
# #"""⼩批量随机梯度下降"""
#     with torch.no_grad():
#         for param in params:
#             param -= lr * param.grad / batch_size
#             param.grad.zero_()
# lr = 0.03
# num_epochs = 3
# net = linreg
# loss = squared_loss
# for epoch in range(num_epochs):#取三个小批量
#     for X, y in data_iter(batch_size, features, labels):
#         l = loss(net(X, w, b), y) # X和y的⼩批量损失
# # 因为l形状是(batch_size,1)，⽽不是⼀个标量。l中的所有元素被加到⼀起，
# # 并以此计算关于[w,b]的梯度
#         l.sum().backward()
#         sgd([w, b], lr, batch_size) # 使⽤参数的梯度更新参数
#     with torch.no_grad():
#         train_l = loss(net(features, w, b), labels)
#         print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
# print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
# print(f'b的估计误差: {true_b - b}')
#-----------------------------------------------------------------------------
# import torch
# import torchvision
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from torch.utils import data
# from torchvision import transforms
# from d2l import torch as d2l
# d2l.use_svg_display()
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0〜1之间
# trans = transforms.ToTensor()
# mnist_train = torchvision.datasets.FashionMNIST(
# root="../data", train=True, transform=trans, download=True)
# mnist_test = torchvision.datasets.FashionMNIST(
# root="../data", train=False, transform=trans, download=True)
# x=DataLoader(mnist_train,batch_size=1,shuffle=False)
# for img,label in x:
#     print(img,label)
#     break
# print(mnist_train[0][0].shape)
# print(mnist_train[0][1])
# def get_fashion_mnist_labels(labels): #@save
#     """返回Fashion-MNIST数据集的⽂本标签"""
#     text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
#     'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
#     return [text_labels[int(i)] for i in labels]
#
# def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5): #@save
#     """绘制图像列表"""
#     figsize = (num_cols * scale, num_rows * scale)
#     _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
#     axes = axes.flatten()
#     for i, (ax, img) in enumerate(zip(axes, imgs)):
#         if torch.is_tensor(img):
#             # 图⽚张量
#             ax.imshow(img.numpy())
#         else:
#             # PIL图⽚
#             ax.imshow(img)
#             ax.axes.get_xaxis().set_visible(False)
#             ax.axes.get_yaxis().set_visible(False)
#         if titles:
#             ax.set_title(titles[i])
#     return axes
# X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
# d2l.plt.show()
#----------------------------------------------------------------------------------
# import torch
# print(torch.__version__)
#--------------------------
# import torch
#
# # 创建一个输入张量 Example_tensor，假设它有3个通道和大小为(32, 32)的图像
# input_channels = 3
# image_size = (32, 32)
# Example_tensor = torch.randn(1, input_channels, image_size[0], image_size[1])
#
# # 设置填充值
# padding_value = 1
#
# # 在输入张量的边界使用非零填充值
# padded_tensor = torch.nn.functional.pad(Example_tensor, (padding_value, padding_value, padding_value, padding_value), value=padding_value)
#
# # 创建一个二维卷积层，并使用填充后的张量进行卷积操作
# conv = torch.nn.Conv2d(in_channels=input_channels,
#                        out_channels=64,
#                        kernel_size=3,
#                        stride=1,
#                        padding=0,  # 因为已经进行了填充，这里设为0
#                        dilation=1,
#                        groups=1,
#                        bias=True)
#
# # 将填充后的张量通过卷积层进行前向传递
# output = conv(padded_tensor)
#
# print(output.shape)
# import numpy as np
#
# data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#
# # 计算中位数
# median = np.percentile(data, 50)
# print("中位数：", median)
#
# # 计算第25和第75百分位数
# percentiles = np.percentile(data, [30, 75])
# print("第25百分位数：", percentiles[0])
# print("第75百分位数：", percentiles[1])
# import torch
# a=torch.randn(1,1,1,1)
# b=torch.randn(1,1,1,1)
# print(a)
# print(b)
# print(torch.mean(a+b))

# import torch
#
# def _fspecial_gauss_1d(size, sigma):
#     coords = torch.arange(size, dtype=torch.float)
#     coords -= size // 2
#
#     g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
#     g /= g.sum()
#
#     return g.unsqueeze(0).unsqueeze(0)
#
# size = 5
# sigma = 1.0
#
# result = _fspecial_gauss_1d(size, sigma)
# print(result)
import torch
import torch.nn as nn

# # 定义一个简单的神经网络模型
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.fc1 = nn.Linear(10, 5)
#         self.fc2 = nn.Linear(5, 2)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.fc2(x)
#         return x
#
# # 创建模型实例
# model = SimpleModel()
#
# # 打印每层的名字和参数
# for name, param in model.named_parameters():
#     print(f"Layer: {name}")
#     print(f"Parameter shape: {param.shape}")
#     print(f"Parameter values:\n{param.data}\n")

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 生成一些示例数据
# np.random.seed(42)
# x = np.random.rand(100, 1)  # 输入数据
# y = 2 * x + 1 + 0.1 * np.random.randn(100, 1)  # 对应的目标输出，模拟带有噪声的线性关系
#
# # 转换为PyTorch张量
# x_tensor = torch.tensor(x, dtype=torch.float32)
# y_tensor = torch.tensor(y, dtype=torch.float32)
#
# # 定义简单的线性回归模型
# class SimpleLinearRegression(nn.Module):
#     def __init__(self):
#         super(SimpleLinearRegression, self).__init__()
#         self.linear1 = nn.Linear(1, 1)  # 一个输入特征，一个输出特征
#         self.rule=nn.ReLU()
#         self.linear2 = nn.Linear(1, 1)
#         self.elu=nn.ELU()
#     def forward(self, x):
#         return self.elu(self.rule(self.linear2(self.rule(self.linear1(x)))))
#
#
# # 初始化模型、损失函数和优化器
# model = SimpleLinearRegression()
# criterion = nn.MSELoss()  # 使用均方误差损失
# optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器
#
# #训练模型
# num_epochs = 100
# for epoch in range(num_epochs):
#     # 前向传播
#     y_pred = model(x_tensor)
#
#     # 计算损失
#     loss = criterion(y_pred, y_tensor)
#
#     # 反向传播和优化
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if (epoch + 1) % 100 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
#
# # 可视化拟合结果
# plt.scatter(x, y, label='Original data')
# plt.plot(x, model(x_tensor).detach().numpy(), 'r', label='Fitted line')
# plt.legend()
# plt.show()

# import torch
#
# # 定义张量
# condition = torch.tensor([[True, False], [False, True]])
# x = torch.tensor([[1, 2], [3, 4]])
# y = torch.tensor([[5, 6], [7, 8]])
#
# # 使用 torch.where
# result = torch.where(condition, x, y)
#
# print(result)
# VOC_COLORMAP = torch.tensor([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
# [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
# [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
# [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
# [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
# [0, 64, 128]])
# print(VOC_COLORMAP.shape)
# for i, colormap in enumerate(VOC_COLORMAP):
#     print(colormap[0])
#     print('saedfgsefgsweg')
# import numpy as np
# arr = np.ones([1000])
# print(arr)









