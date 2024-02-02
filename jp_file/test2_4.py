# import torch
# from torch import nn as nn
# class qq(nn.Sequential):
#     def __init__(self,in_channel,out_channel):
#         super(qq, self).__init__(nn.Linear(in_channel,out_channel,bias=False),nn.Softmax(dim=1))
# q1=qq(6,2)
# print(q1)
# xx=torch.randn(1,2,3)
# xx=torch.flatten(xx,start_dim=1)
# print(xx)
# x=q1(xx)
# print(x)
#-----------------------------
# import numpy as np
# inverted_residual_setting = [
#             # t, c, n, s
#             [1, 16, 1, 1],
#             [6, 24, 2, 2],
#             [6, 32, 3, 2],
#             [6, 64, 4, 2],
#             [6, 96, 3, 1],
#             [6, 160, 3, 2],
#             [6, 320, 1, 1],
#         ]
# inverted_residual_setting=np.array(inverted_residual_setting)
# print(inverted_residual_setting.shape)
# x=0
# for t,c,n,s in inverted_residual_setting:
#     print(t)
#     print(c)
#     print(n)
#     print(s)
#     x+=1
#     print('第{}次'.format(x))
#------------------------------------------
# from torch import nn
# import torch
# a=torch.randn(3,4,5)
# print(a)
# b,c=a.shape[:2]
# print(b)
# print(c)
# print(a.shape)
# b=torch.arange(2*3*4*5)
# b=b.reshape(2,3,4,5)
# print(b)
# print(b.shape)
# c=a*b
# print(c.shape)
# print(c)
# x=torch.mean(c,dim=1)
# print(x)
# print(x.shape)
#----------------------------------------
# import torch                  #注意力机制
# import torch.nn as nn
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)#通道数不变
#         self.max_pool = nn.AdaptiveMaxPool2d(1)#通道数不变
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction_ratio),
#             nn.ReLU(),
#             nn.Linear(in_channels // reduction_ratio, in_channels)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.avg_pool(x)
#         max_out = self.max_pool(x)
#         avg_out = avg_out.view(avg_out.size(0), -1)
#         max_out = max_out.view(max_out.size(0), -1)
#         avg_weight = self.fc(avg_out)
#         max_weight = self.fc(max_out)
#         channel_attention = self.sigmoid(avg_weight + max_weight).unsqueeze(2).unsqueeze(3).expand_as(x)
#         return x * channel_attention
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), "kernel size must be 3 or 7"
#         padding = 3 if kernel_size == 7 else 1
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)#步长默认为1
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)#以通道进行拼接
#         x = self.conv(x)
#         spatial_attention = self.sigmoid(x)
#         return x * spatial_attention
#
#
# class AttentionModule(nn.Module):
#     def __init__(self, in_channels):
#         super(AttentionModule, self).__init__()
#         self.channel_attention = ChannelAttention(in_channels)
#         self.spatial_attention = SpatialAttention()
#
#     def forward(self, x):
#         x = self.channel_attention(x)
#         x = self.spatial_attention(x)
#         return x
#--------------------------------------------------------------------------------
# import torch
# import torch.nn as nn
#
# # 定义一个转置卷积层进行上采样
# upsample = nn.ConvTranspose2d(3, 6, 3, stride=2, padding=1)
#
# # 定义一个输入张量
# x = torch.randn(1, 3, 32, 32)
#
# # 对输入张量进行上采样
# out = upsample(x)
#
# # 输出上采样后张量的大小
# print(out.size())
#------------------
# class x():
#     def __init__(self,i):
#         print(i)
#     def __call__(self,x):
#         print("调用了")
# from torch import nn as nn
# nn.BatchNorm2d()
#---------------------------------===========
# import os
# cpu_count = os.cpu_count()
# print("CPU核心数：", cpu_count)#输出为8
#--------------------
# import torch
# x=[(1,2),(3,4),(5,6)]
# print(*x)
# print(list(zip(*x)))
# a,b=list(zip(*x))
# print(a)
# print(b)
#------------------------
# import torch
# from torchvision.datasets import ImageFolder
# dataset = ImageFolder("D:\pytorch_\PyTorch_tudui\hymenoptera_data\\train", transform=None) #主要有两个参数，一个是图像根目录（被映射成标签的子目录的上一级），一个是数据操作
# train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
# print(dataset[125][0].size) #第一张图片的图片矩阵
# print(dataset[125][1]) #第一张图片的标签
#--------------------------------------------------------------------------
# import torch
# import torch.nn.functional as F
# # 创建一个输入张量
# input_tensor = torch.randn(1, 3, 32, 32)
# # 将输入张量调整为指定大小
# output1 = F.interpolate(input_tensor, size=(64, 64), mode='nearest')
# # 将输入张量按比例缩放
# output2 = F.interpolate(input_tensor, scale_factor=2, mode='bilinear')
# # 打印插值后的结果张量大小
# print(output1.size())
# print(output2.size())


