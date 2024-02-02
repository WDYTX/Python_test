# from torch.utils.tensorboard import SummaryWriter
# writer=SummaryWriter('log')
# for i in range(100):
#     writer.add_scalar('y=3x',2*i,i)
# writer.close()
#-----------------------------------------------
# import torch
# import torchvision
# from PIL import Image
# from torch.utils.tensorboard import SummaryWriter
# writer=SummaryWriter('log')
# path1="D:/pytorch_/PyTorch/hymenoptera_data/train/ants/6240338_93729615ec.jpg"
# path2="D:/pytorch_/PyTorch/hymenoptera_data/train/ants/0013035.jpg"
# img1=Image.open(path1)
# img2=Image.open(path2)
# tensor=torchvision.transforms.ToTensor()
# img1=tensor(img1)
# img2=tensor(img2)
# writer.add_image('image',img1,global_step=1)
# writer.add_image('image',img2,global_step=2)
# writer.close()
#----------------------------------------
# from PIL import Image
# from torchvision import transforms
# from torch.utils.tensorboard import SummaryWriter
# writer=SummaryWriter('log')#类的实例化并创建一个文件夹
# #读入图片信息
# img=Image.open("D:/download/hymenoptera_data/train/ants/0013035.jpg")
# img.show()
# transform=transforms.ToTensor()#创建一个实例对象
# #将图片信息转换为张量
# img=transform(img)
# writer.add_image("test",img)
# writer.close()
#------------------------------
# import torch
# import torchvision
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from torch.utils import data
# from torchvision import transforms
# from d2l import torch as d2l
# writer=SummaryWriter('log')
# d2l.use_svg_display()
# # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# # 并除以255使得所有像素的数值均在0〜1之间
# trans = transforms.ToTensor()
# i=0
# mnist_train = torchvision.datasets.FashionMNIST(
# root="../data", train=True, transform=trans, download=True)
# mnist_test = torchvision.datasets.FashionMNIST(
# root="../data", train=False, transform=trans, download=True)
# x=DataLoader(mnist_train,batch_size=64,shuffle=False)
# for img,label in x:
#     writer.add_image("mnist_img", img, global_step=i, dataformats="NCHW")
#     i+=1
# writer.close()
# #---------------------------------------------------------------------
# from tqdm import tqdm
# import time
# # 创建一个列表作为示例
# my_list = range(10)
#
# # 使用 tqdm 包装迭代对象
# for item in tqdm(my_list, desc="Processing"):
#     # 模拟处理时间
#     time.sleep(0.5)
#------------------------------------------------------------------------
# import torch
# import torch.nn as nn
# dropout = nn.Dropout(p=0.5).train() # dropout概率为0.5
# input = torch.tensor([[1., 2.],[1.,2.]])  # 输入张量
# print(input)
# output = dropout(input)  # Dropout操作
# print(output)
# dropout = nn.Dropout(p=0.5).eval() # dropout概率为0.5
# input = torch.tensor([[1., 2.],[1.,2.]])  # 输入张量
# output = dropout(input)
# print(output)
#---------------------------------------------------------------------
# import torch
# a=torch.tensor([[1,2,344,123,12],[3,3455,677,74,666]])
# b=torch.max(a,dim=1)[1].data.numpy()
# print(b)
#----------------------------------------
# import torch
# import torch.nn as nn
# # 定义一个Sequential容器
# model = nn.Sequential(
#     nn.Linear(1, 1),  # 全连接层：输入特征数为1，输出特征数为6
#     nn.ReLU(),        # ReLU激活函数
#     nn.Linear(1, 2), # 全连接层：输入特征数为6，输出特征数为10
#     nn.Softmax(dim=1) # Softmax函数，dim=1表示按行计算Softmax
# )
# # 打印模型结构
# print(model)
# a = torch.tensor([6.0])
# a = torch.unsqueeze(a, dim=0) # 将输入张量的维度从1变为2,batch_size=1,
# b = model(a)
# print(b)
#--------------------------------------------------------------------------
# import torch
#
# # 创建一个大小为(2, 3, 4)的张量
# x = torch.randn(2, 3, 4, 5)
# print("原始张量:\n", x)
# 将张量展平为一维
# flattened = torch.flatten(x,start_dim=1)
# print("展平后的张量:\n", flattened)
# print(flattened.shape)
#-------------------------
# def my_generator():
#     yield 1
#     yield 2
#     yield 3
# # 使用生成器函数
# gen = my_generator()
# print(next(gen))  # 输出：1
# print(next(gen))  # 输出：2
# print(next(gen))  # 输出：3迭代完成
#--------------------------------------------------------------------
# import torch
# a=torch.randn(2,5)
# print(a)
# b=torch.squeeze(a)
# c=torch.softmax(b,dim=1)
# d=torch.softmax(b,dim=0)
# print(b)
# print(c)
# print(d)
#--------------------------------------------
# import torch.nn as nn
# # 定义一个示例的神经网络模型
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(10, 20)
#         self.fc2 = nn.Linear(20, 10)
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.fc2(x)
#         return x
# # 创建一个示例模型实例
# net = Net()
# # 获取模型的状态字典
# model_state_dict = net.state_dict()
# print(model_state_dict)#输出模型参数
#--------------------------------------------------
# import torch
# import torch.nn as nn
# class CustomModel(nn.Module):
#     def __init__(self):
#         super(CustomModel, self).__init__()
#         self.layer1 = nn.Linear(10, 20)
#         self.layer2 = nn.Linear(20, 10)
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         return x
# model = CustomModel()
# # 迭代模型的所有子模块并输出名称
# for module in model.modules():
#     print(module.__class__.__name__)
#---------------------------------------------------
# import torch
# a=torch.randn(1,2,1)
# print(a)
# b=torch.softmax(a,dim=0)
# print(b)
# c=torch.softmax(a,dim=1)
# print(c)
#-----------------------------------
# import torch
# import torch.nn as nn
#
# # 定义一个含有批归一化的神经网络模型
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(784, 256)
#         self.bn1 = nn.BatchNorm1d(256)  # 批归一化层
#         self.fc2 = nn.Linear(256, 128)
#         self.bn2 = nn.BatchNorm1d(128)  # 批归一化层
#         self.fc3 = nn.Linear(128, 10)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = torch.relu(x)
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = torch.relu(x)
#         x = self.fc3(x)
#         return x
# # 创建模型实例
# model = Net()
# # 创建输入数据（假设为随机的张量）
# x = torch.randn(100, 784)
# # 前向传播
# output = model(x)
# # 输出结果
# print(output)
#-----------------------------------
# import torch
# a=torch.randn(1,2,3,1)
# b=torch.randn(1,2,3,1)
# print('a为',a)
# print('b为',b)
# c=torch.cat((a,b),dim=2)
# print(c)
# print(c.shape)
#-----------------------------------------------------------------------------
# import torch
# from torch import nn as nn
# y=nn.MaxPool2d(kernel_size=(3,3),stride=2)
# a=torch.randn(5,5)
# a=a.unsqueeze(0).unsqueeze(0)
# print(a)
# print(a.shape)
# b=y(a)
# print(b.shape)
# print(b)
#------------------------------------------







