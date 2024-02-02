# import torch
#
# tensor_a = torch.tensor([1, 2, 3])   # 形状为 (3,)
# tensor_b = torch.ones((2, 3))        # 形状为 (2, 3)
#
# expanded_tensor_a = tensor_a.expand_as(tensor_b)
# print(tensor_a.shape)
# print(expanded_tensor_a)
# print(expanded_tensor_a.shape)
#-------------------------------------------------------
# from torch import nn as nn
# import torch
# a=torch.randn(4,21,480,480)
# b=torch.randint(0,21,(4,480,480))
# print('a的形状',a.shape)
# print('b之前的形状',b.shape)
# print('b目前形状',b.shape)
# cc=nn.CrossEntropyLoss()
# loss=cc(a,b)
# print(loss.item())
#-----------------------------------------------------------
# import torch
# import torch.nn as nn
# # 假设有4个样本的类别索引标签
# labels = torch.tensor([1, 2, 0, 1])
# # 假设模型预测的结果为概率分布形式，需要通过softmax函数将其转换为概率值
# predictions = torch.tensor([[0.2, 0.6, 0.2],
#                             [0.1, 0.2, 0.7],
#                             [0.8, 0.1, 0.1],
#                             [0.3, 0.4, 0.3]])
# # 创建交叉熵损失函数
# loss_fn = nn.CrossEntropyLoss()
# # 计算损失
# loss = loss_fn(predictions, labels)
# print(loss)
# p=predictions.argmax(dim=1)
# print(p)
#--------------------------------------------------------
# from torch import nn as nn
# import torch
# a=torch.randn(1,2,1,1)
# print(a)
# b=torch.randint(0,2,(1,1,1))
# print(b)
# print('a的形状',a.shape)
# print('b之前的形状',b.shape)
# print('b目前形状',b.shape)
# cc=nn.CrossEntropyLoss()
# loss=cc(a,b)
# print(loss.item())
#----------------------------------------------------------------
# import numpy as np
# from PIL import Image
#
# # 创建一个3通道的随机数组
# array = np.random.randint(0, 256, (600, 600, 3), dtype=np.uint8)
#
# # 将数组转换为图像对象
# image = Image.fromarray(array)
# print(image)
# # 显示图像
# image.show()
#-------------------------------------------
# import torch
# x=[[0.1,0.6,0.3],[0.3,0.3,0.4],[0.3,0.2,0.5],[0.8,0.05,0.05]]
# x1=[[0,1.,0],[1.,0,0],[1.,0,0],[0,1.,0]]
# print('之前的x:',x)
# x=torch.tensor(x)
# print('之后的x',x)
# print('之前的x1:',x1)
# x1=torch.tensor(x1)
# print('之后的x',x1)
# y=[1,0,0,1]
# print('之前的y:',y)
# y=torch.tensor(y)
# print('之后的y:',y)
# y1=[2,1,1,2]
# print('之前的y1:',y1)
# y1=torch.tensor(y1)
# print('之后的y1:',y1)
# loss=torch.nn.CrossEntropyLoss()
# l1=loss(x,y)
# l2=loss(x,y1)
# l3=loss(x1,y)
# l4=loss(x1,y1)
# print('l1损失为：',l1)
# print('l2损失为：',l2)
# print('l3损失为：',l3)
# print('l4损失为：',l4)
#--------------------------------------------------
# import torch
# x=[[0.1,0.6,0.3],[0.3,0.3,0.4],[0.3,0.2,0.5],[0.8,0.05,0.05]]
# x1=[[0,100.,0],[100.,0,0],[100.,0,0],[0,100.,0]]
# x1=torch.tensor(x1)
# y=torch.tensor([1,0,0,1])
# loss=torch.nn.CrossEntropyLoss()
# l1=loss(x1,y)
# print('l1损失为：',l1)
#---------------------------------------------------------------
# import torch
#
# a=torch.randn(1,2,1,1)
# print(a)
# b=torch.nn.functional.interpolate(a,(4,4),mode='bilinear',align_corners=False)
# print(b.shape)
# print(b)
# c=b=torch.nn.functional.interpolate(a,scale_factor=(4,3),mode='bilinear',align_corners=False)
# print(c.shape)
# print(c)
#-----------------------------------------------------------
# import torch
# import torch.nn as nn
# class conv(nn.Module):
#     def __init__(self, ch_in=3, ch_out=3):
#         super(conv, self).__init__()
#         self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1,dilation=2, bias=True)
#     def forward(self, x):
#         x = self.conv1(x)
#         return x
# t = conv()
# a=torch.randn(32,3,224,224)
# print(t(a).shape)
# 3---------------
# import torch
# x=torch.randint(0,10,(1,4,3,2))
# print(x)
# x=x.view(1,2,2,3,2)
# x=torch.transpose(x,1,2).contiguous()
# print(x.view(1,-1,3,2))
import numpy as np
import torch
#-----------------------------------
# import random
# random.seed(0)
# # 原始数据列表
# numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#
# # 验证集占总数据的比例
# val_rate = 0.3
#
# # 计算需要抽取的元素数量
# val_size = int(len(numbers) * val_rate)
#
# # 随机抽取元素构建验证集
# val_set = random.sample(numbers, k=val_size)
#
# # 剩余部分作为训练集
# train_set = [num for num in numbers if num not in val_set]
#
# print("原始数据列表:", numbers)
# print("验证集:", val_set)
# print("训练集:", train_set)
#----------------------------------------
# images = ['img1', 'img2']
# targets = ['target1', 'target2']
# batch = list(zip(images, targets))
# print(batch)
# bb,cc=zip(*batch)
# print(bb)
# print(cc)

#----------------------------
# class a:
#     def __init__(self,x):
#         self.x=x
#     def __call__(self,y):
#         print(self.x)
# A=a(6)
# A(2)
#---------------------------------------------
# from PIL import Image
# from torchvision import transforms
# from torchvision.transforms import functional as f
# a=Image.open('D:\Pascal_dataset\VOCdevkit\VOC2012\SegmentationClass\\2007_000323.png')
# # b = torch.as_tensor(np.array(a), dtype=torch.int64)
# # a=np.array(a)
# #b=transforms.ToTensor()#像素值范围会缩放到[0, 1]之间
# c=b(a)
# print(a)
# print(c)
#-------------------------------------------------------
# from PIL import Image
# from torchvision import transforms
# from torchvision.transforms import functional as f
# a=Image.open('D:\Pascal_dataset\VOCdevkit\VOC2012\JPEGImages\\2007_000033.jpg')
# b = torch.as_tensor(np.array(a), dtype=torch.int64)
# print(b.shape)
# print(b)
#------------------------------------------------------------------------
# import torch
# from torch.nn import functional as F
# x=torch.rand(1,3,4,4)
# y=torch.nn.AvgPool2d((2,2),1,0)
# cc=F.avg_pool2d(x,1,2)
# cc.shape
# print(cc)
#------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# 定义一个简单的卷积神经网络
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = datasets.CIFAR10(root='../data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
# 初始化模型和优化器
num_classes = 10  # CIFAR-10有10个类别
model = ConvNet(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(10):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

print("Training finished.")

# 在测试集上评估模型
test_dataset = datasets.CIFAR10(root='../data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy:.2f}')











