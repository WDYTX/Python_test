# import torch
# a = torch.tensor([0, 1, 1, 2, 2, 3, 1, 2, 3])#真实值
# b = torch.tensor([1, 1, 1, 2, 3, 3, 1, 2, 3])#预测值
# k = (a >= 0) & (a < 4)#类别范围
# print(k)
# inds = 4 * a[k] + b[k]
# print(inds)
# mat = torch.zeros(4, 4)
# mat += torch.bincount(inds, minlength=3**2).reshape(4, 4)#统计出现的次数
# h = mat.float()
# # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
# acc_global = torch.diag(h).sum() / h.sum()
# # 计算每个类别的准确率
# acc = torch.diag(h) / h.sum(1)
# # 计算每个类别预测与真实目标的iou
# iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
# print(mat)
# print(acc_global)
# print(acc)
# print(iu)
# ---------------------------------------------------------------
# import torch
# from torch import nn
# class xx(nn.Module):
#     def __init__(self):
#         super(xx, self).__init__()
#         self.scale=2**-0.5
#         self.l1=nn.Linear(2,2)
#         self.l2=nn.Linear(2,2)
#         self.l3=nn.Linear(2,6)
#     def forward(self,x):
#         q=self.l1(x)
#         k=self.l2(x)
#         v=self.l3(x)
#         att=(q @ k.transpose(-2,-1))*self.scale
#         y=k.transpose(-2,-1)
#         t=att.softmax(dim=-1)
#         print(t)
#         x=t@v
#         return x
# a=torch.rand(1,4,3,2,dtype=torch.float32)
# print(a.shape)
# print(a)
# c=xx()
# y=c(a)
# print(y.shape)
# print(y)
# ----------------------------------------------
# import torch
# a=torch.rand(1,3,4,4,dtype=torch.float32)
# print(a)
# b=torch.randint(1,3,(1,4,4))
# print(b)
# loss=torch.nn.functional.cross_entropy(a,b)
# print(loss)
import numpy as np
import torch

# -----------------------------------------------------------------------------
# import torch
# a=torch.randint(0,17,(3,3))
# print(a)
# print(a.shape)
# x=torch.tensor([0,1,2,7,3,4,6,3,2,55,55,53,65,58,53,6,1,2,0,4,546])
# b=x[a]
#
# print(b)
# print(b.shape)

#-------------------------------
# VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
# [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
# [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
# [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
# [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
# [0, 64, 128]]
#
# VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
# 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
# 'diningtable', 'dog', 'horse', 'motorbike', 'person',
# 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
#
# def voc_colormap2label():
#     """构建从RGB到VOC类别索引的映射"""
#     colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
#     #通过构建256*256*256的一维张量以便RGB即使都取得255能够刚好映射到最后一位
#     for i, colormap in enumerate(VOC_COLORMAP):# 通过VOC_COLORMAP迭代依次将其转变为相应的类别
#         #例如[0,0,0]将对应colormap2label第0个元素，[128, 0, 0]将对应128*256*256个元素且该值1，以此类推，通过循环来进行的，使得每个RGB值都能唯一对应
#         colormap2label[
#             (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
#     return colormap2label#对于该例子中的VOC_COLORMAP大部分为0
# #@save
# def voc_label_indices(colormap, colormap2label):
#     """将VOC标签中的RGB值映射到它们的类别索引"""
#     colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
#     idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
#             + colormap[:, :, 2])
#     #对于一张RGB图像来说，通道0是红色通道（R通道），通道1是绿色通道（G通道），通道2是蓝色通道（B通道）
#     #通过该行代码可以使得该图片每个像素都有一个相应的编号
#     return colormap2label[idx]#通过这个编号索引出对于的类别(得到的类别是一个数字如0,1,2等)
#-------------------------------------------------------------------------
# import torch
# from PIL import Image
# from torchvision.transforms import functional as F
# a=Image.open('../data/1/0013035.jpg')
# a=torch.as_tensor(np.array(a), dtype=torch.int64)
# print(a)
# b=Image.open('D:\Pascal_dataset\VOCdevkit\VOC2012\JPEGImages\\2007_001288.jpg')
# b=torch.as_tensor(np.array(b), dtype=torch.int64)
# print(b)
# c=Image.open('D:\Pascal_dataset\VOCdevkit\VOC2012\SegmentationClass\\2007_000129.png')
# c = torch.as_tensor(np.array(c), dtype=torch.int64)
# print(c)
#---------------------------------------------------------------------------------------------------------------------------
# import torch
#
# # 创建两个示例张量
# tensor1 = torch.tensor([1, 2, 3, 4])
# tensor2 = torch.tensor([1, 0, 3, 4])
# x=torch.tensor([1,2,3,4])
# c=0
# # 执行元素级别的不等于比较
# result = torch.ne(tensor1, tensor2)
# jpg= torch.ne(x,c)
# print(jpg)
# x=x[result]
# print(x)
# x=torch.arange(0,1000)
# xx=torch.tensor([[1,2,34,5,6,7,8,10,100],[2,3,54,7,9,9,6,344,3]])
# y=x[xx]
# print(y)
#------------------------------------------------------------------------
# import torch
# class cc():
#     def __init__(self):
#         pass
#     def max(self):
#         print('niubi')
#     @property
#     def value(self):
#         print('sadfasdfa')
# c=cc()
# c.max()
# c.value
#-----------------------
# import torch
# a=torch.randint(1,10,(3,6,9))
# print(a.shape)
# b=a[:,0]
# print(b.shape)
#-------------------------------------------------------------
# import torch
# import torch.nn as nn
# class xx(nn.Module):
#     def __init__(self):
#         super(xx,self).__init__()
#         self.conv1=nn.Conv2d(3,6,3,2,1,2)
#     def forward(self,x):
#         return self.conv1(x)
# y=xx()
# x=y(torch.randn(1,3,224,224))
# print(x.shape)
# import torch
# import torch.nn.functional as F
#
# x = torch.tensor([[1, 2],
#                   [3, 4]])
#
# # 在四个方向上都填充1个元素，填充值为0
# padded_x = F.pad(x, (1, 2, 1, 1), mode='constant', value=0)
#
# print(padded_x)
#-----------------------------------------------------
# import torch
# def drop_path_f(x, drop_prob: float = 0., training: bool = False):
#     """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
#
#     This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
#     the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
#     See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
#     changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
#     'survival rate' as the argument.
#
#     """
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets #创建一个与输入张量 x 形状相同，但是除了第一个维度外，其余维度大小都为1的新形状元组
#     random_tensor = keep_prob + torch.rand(shape, dtype=torch.float)
#
#     random_tensor.floor_()  # binarize  # 对张量中的每个元素进行向下取整操作
#     print(random_tensor)
#     output = x.div(keep_prob) * random_tensor#以广播机制使得某些层全为0，也就是让某些层失活
#     print(output)
#     return output
#
# x=torch.randint(1,5,(4,3,5,5))
# print(x)
# y=drop_path_f(x,0.5,True)
# print(y)
#--------------------------------------------------------------------------------------------------------
# import torch
# import torch.nn.functional as F
#
# # 创建一个2x3的张量
# x = torch.tensor([[1, 2, 3], [4, 5, 6]])
#
# # 在第一个维度上填充1个元素，在第二个维度上填充2个元素
# padded_x = F.pad(x, (1, 2))
#
# print(padded_x)
#------------------------------------------------------
# import torch
# from torch.nn import functional as F
# x=torch.randn(2,3,2,2)
# print(x)
# y=F.pad(x,(0,2,0,1,0,0))
# print(y)
# import torch
# from torch import nn as nn
#
# class xx(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.t=5
#     def forward(self,x):
#         h=self.h
#         print(x)
#         print(h)
# b=xx()
# b.h=10
# b(5)
#
# import torch
#
# x = torch.tensor([1, 2, 3, 4, 5])
# mask = torch.tensor([True, False, True, False, True])
#
# x_masked = x[mask]
# x_masked = x_masked * 2
#
# x[mask] = x_masked
#
# print(x)

#----------------------------
# i = 100
# l1 = []
# for i in range(10):
#     l1.append(i)
# j = 100
# l2 =[j for j in range( 10)]
# print(i,j)

#---------------------------------------------------
# import cv2
# import numpy as np
#
# # 读取两个输入图像
# image1 = cv2.imread('D:\pytorch_\PyTorch_tudui\hymenoptera_data\\train\\bees\\16838648_415acd9e3f.jpg')
# image2 = cv2.imread('D:\pytorch_\PyTorch_tudui\hymenoptera_data\\train\\bees\\154600396_53e1252e52.jpg')
# new_width = 900
# new_height = 600
# image1=cv2.resize(image1, (new_width, new_height))
# image2=cv2.resize(image2, (new_width, new_height))
# # 确保两个输入图像具有相同的尺寸
# if image1.shape != image2.shape:
#     raise ValueError("输入图像的尺寸不一致")
#
# # 设置融合权重（可以根据需要进行调整）
# alpha = 0.5  # 第一个图像的权重
# beta = 1  # 第二个图像的权重
#
# # 图像融合
# blended_image = cv2.addWeighted(image1, alpha, image2, beta, 0.0)
#
# # 显示融合结果
# cv2.imshow('Blended Image', blended_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#---------------------------------------------------------------------------------

# class MyClass:
#     def __init__(self):
#         self._my_variable = 42
#         print('_my_variable')
#     def _my_function(self):
#         print("This is a protected function.")
#
#     def __my_method__(self):
#         print("This is a name-mangled method.")
#
# class MySubclass(MyClass):
#     def __init__(self):
#         super().__init__()
#
#     def call_methods(self):
#         self._my_function()      # 可以访问受保护的函数
#         self.__my_method__()    # 通过名称修饰访问父类的方法
#
# obj = MySubclass()
# # obj.call_methods()
#------------------------------------------
# import matplotlib.pyplot as plt
# import numpy as np
#
# def gelu(x):
#     return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
#
# x = np.linspace(-10, 10, 100)
# y = gelu(x)
#
# plt.figure(figsize=(8, 4))
# plt.plot(x, y, label='GELU', linewidth=2)
# plt.title('GELU Activation Function')
# plt.xlabel('Input')
# plt.ylabel('Output')
# plt.grid(True)
# plt.legend()
# plt.show()
#---------------------------------------------------------
# import torch
# a=torch.arange(3)
# b=torch.arange(3)
# print(a)
# print(b)
# dd,ee=torch.meshgrid(a,b,indexing='ij')
#
# cc=torch.stack(torch.meshgrid(a,b,indexing='ij'))
# print(cc)
#
# print(dd,dd.shape)
# print(ee)
#---------------------------------------------------------
# import torch
# a=torch.tensor([[ 0,0,1,1],[ 0,1,0,1]])
# print(a)
# a=a[:,None,:]
# b=a[:,:,None]
# print(b)
# print(a)
# print(b-a)
#----------------------------------------------------------------------
# from torch import nn as nn
# class CA_Block(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(CA_Block, self).__init__()
#
#         self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
#                                   bias=False)
#
#         self.relu = nn.ReLU()
#         self.bn = nn.BatchNorm2d(channel // reduction)
#
#         self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
#                              bias=False)
#         self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
#                              bias=False)
#
#         self.sigmoid_h = nn.Sigmoid()
#         self.sigmoid_w = nn.Sigmoid()
#
#     def forward(self, x):
#         _, _, h, w = x.size()
#
#         x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
#         x_w = torch.mean(x, dim=2, keepdim=True)
#
#         x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
#
#         x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)#在最后一个维度上将其按照[h,w]进行划分
#
#         s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
#         s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
#
#         out = x * s_h.expand_as(x) * s_w.expand_as(x)
#         return out
# ca=CA_Block(16)
# a=torch.randn(1,16,24,24)
# out=ca(a)
# print(a)
#-----------------------------------------------------------------------
# import torch
# a=torch.randn(1,1,3)
# b=torch.randn(8,3,3)
# print(a)
# print(b)
# c=a+b
# print(c)
# print(a.shape)
# print(b.shape)
# print(c.shape)
# d=a.expand(b.shape[0],-1,-1,None)
# print(d.shape)

#-----------------------------------------------------------
# import torch
#
# # a=[1,2,3,4,5,6,78,90,999]
# # dpr=[1,2,3,4,5,6,7,8,9,10]
# # b=dpr[sum(a[:1]):sum(a[:2])]
# # print(b)
# #
# # print(sum(a[:0]))
#
# import torch
#
# drop_path_rate = 0.1  # 示例的 drop_path_rate
# depths = [3, 4, 2]  # 示例的 depths 列表，包含3个元素
#
# # 使用 torch.linspace 创建一个等差数列，范围从0到 drop_path_rate，总共包含 depths 中所有元素数量的值
# linspace_values = torch.linspace(1, 10, sum(depths))
#
# # 使用列表推导将每个张量元素（标量）提取出来，并存储在列表中
# values_list = [x.item() for x in linspace_values]
#
# # 打印列表中的值
# print(values_list)

# import torch
# a=torch.ones((1,1,1,1))
#
# print(a)


