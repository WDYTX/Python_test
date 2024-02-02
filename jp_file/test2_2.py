# import torch
# import numpy
# #将数据导入xy_data中
# xy_data=numpy.loadtxt('diabetes.csv.gz',delimiter=',',dtype=numpy.float32)#diabetes.csv.gz该文将要放到同一个目录中
# #再将其中的数据分别放入x,y----------注意这个数据是以矩阵的形式存的，最后一列存的是输出也就是y
# #每一行代表的是一个样本
# x=torch.from_numpy(xy_data[:,:-1])
# y=torch.from_numpy(xy_data[:,[-1]])
# #定义模型
# class mm(torch.nn.Module):
#     def __init__(self):
#         super(mm, self).__init__()
#         self.linear1=torch.nn.Linear(8,6)
#         self.linear2=torch.nn.Linear(6,4)
#         self.linear3=torch.nn.Linear(4,1)
#         self.sigmoid=torch.nn.Sigmoid()
#     def forward(self,x):
#         x1=self.sigmoid(self.linear1(x))
#         x2=self.sigmoid(self.linear2(x1))
#         y_hat=self.sigmoid(self.linear3(x2))
#         return y_hat
# #将类实例化---对象
# M=mm()
# #损失
# loss=torch.nn.BCELoss()
# #更新器
# updata=torch.optim.SGD(M.parameters(),lr=0.01)
# #训练---迭代
# for i in range(20000):
#     #求y_hat---使用模型
#     y_hat=M(x)
#     #求损失，调用
#     l=loss(y_hat,y)
#     #梯度清零
#     updata.zero_grad()
#     #求梯度
#     l.backward()
#     #更新参数
#     updata.step()
# test=torch.tensor([[-0.294118,0.487437,0.180328,-0.292929,0,0.00149028,-0.53117,-0.0333333],\
#                    [-0.882353,-0.145729,0.0819672,-0.414141,0,-0.207153,-0.766866,-0.666667],\
#                    [0,0.417085,0.377049,-0.474747,0	,-0.0342771	,-0.69684,	-0.966667],\
# [-0.764706	,0.758794	,0.442623	,0	,0	,-0.317437	,-0.788215,	-0.966667],\
# [-0.764706,	-0.0753769	,-0.147541	,0,	0	,-0.102832	,-0.9462,	-0.966667],\
# [-0.647059,	0.306533	,0.278689	,-0.535354,	-0.813239	,-0.153502,	-0.790777,	-0.566667],\
# [-0.0588235	,0.20603,	0.409836,	0,	0	,-0.153502	,-0.845431	,-0.966667],\
# [-0.764706,	0.748744,	0.442623,	-0.252525,	-0.716312,	0.326379,	-0.514944,	-0.9]
# ])
# output1=M(test)#预测概率
# print(output1)
# row=len(test)
# output1=(output1>((torch.empty(row,1)).fill_(0.5))).int()
# print(output1.tolist())#转换成列表
#----------------------------------------------
# import torch
# a=torch.tensor([[1,2,3,4,5],[1,3,456,67,7]])
# b=torch.tensor([1,1])
# print(a[b])
#---------------------------------------------------------
# import torch
# import torchvision
# batch_size=64
# c=0
# transforms=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(0.1307,0.3081)])#将pil转变为tensor
# #得到相应的数据
# train_data=torchvision.datasets.MNIST(root='../data/MNIST',train=True,download=True,transform=transforms)
# #得到小批量的数据
# batch_train_data=torch.utils.data.DataLoader(train_data,shuffle=True,batch_size=batch_size)
# #得到相应的测试数据---与训练数据一样的方法
# test_data=torchvision.datasets.MNIST(root='../data/MNIST',train=False,download=True,transform=transforms)
# batch_test_data=torch.utils.data.DataLoader(test_data,shuffle=False,batch_size=batch_size)
# #定义模型
# import torch
# class module(torch.nn.Module):
#     def __init__(self):
#         super(module, self).__init__()
#         self.l1=torch.nn.Linear(784,512)
#         self.l2=torch.nn.Linear(512,256)
#         self.l3=torch.nn.Linear(256,128)
#         self.l4=torch.nn.Linear(128,64)
#         self.l5=torch.nn.Linear(64,10)
#     def forward(self,x):
#         x=x.reshape(-1,784)
#         x=torch.nn.functional.relu(self.l1(x))
#         x=torch.nn.functional.relu(self.l2(x))
#         x = torch.nn.functional.relu(self.l3(x))
#         x = torch.nn.functional.relu(self.l4(x))
#         return self.l5(x)
# #类的实例化
# MM=module()
# #损失
# loss=torch.nn.CrossEntropyLoss()#使用交叉熵
# #更新器
# updata=torch.optim.SGD(MM.parameters(),lr=0.01)#使用随机梯度下降来更新参数
# #定义一个训练函数,rpoch表示训练次数
# def train_f():
#     sunshi=0.0
#     global c
#     for x,y in batch_train_data:
#         y_hat=MM(x)
#         l=loss(y_hat,y)
#         updata.zero_grad()
#         l.backward()
#         updata.step()
#         sunshi=sunshi+l.item()
#     c+=1
#     print('小批量迭代次数：{}，最后这次损失为{}'.format(c,sunshi/len(y)))
# def test_f():
#     #需要算出预测正确的值
#     correct=0
#     total=0
#     with torch.no_grad():
#         for x,y in batch_test_data:
#             output=MM(x)
#             _,output=torch.max(output,dim=1)
#             total+=len(y)
#             correct+=((output==y)*1).sum().item()
#     print('Accuracy on test set: %d %%' % (100 * correct / total))#百分号形式
# for i in range(20):
#     train_f()
#     test_f()
#-----------------------------------------------------------------------------
# import torch
#
# # 生成一个形状为[2, 3]的随机数张量
# x = torch.randn(2, 3)
# print(x)
#---------------------------------------------------------------------------
# import torch
# input_channel=10
# output_channel=5
# input_weight=100
# input_height=100
# kernel_size=3
# batch_size=1
# input=torch.randn(batch_size,input_channel,input_weight,input_height)
# #卷积层实例化
# conv_layer=torch.nn.Conv2d(input_channel,output_channel,kernel_size)
# output=conv_layer(input)
# print('输入的值：',input)
# print('输入的形状：',input.shape)
# print('输出的值：',output)
# print('输出的形状：',output.shape)
# print('卷积层权重的形状：',conv_layer.weight.shape)
# print('卷积层权重的值：',conv_layer.weight)
#----------------------------------------------------------------------
# import torch
# input=[3,4,6,5,7,\
#        2,4,6,8,2,\
#        1,6,7,8,4,\
#        9,7,4,6,2,\
#        3,7,5,4,1,\
#       ]
# input=torch.tensor(input,dtype=torch.float32).reshape(1,1,5,5)
# print('输入量为',input)
# conv_layer=torch.nn.Conv2d(1,1,kernel_size=3,padding=1,bias=False)
# kernel=torch.tensor([1.,2.,3.,4.,5.,6.,7.,8.,9.]).reshape(1,1,3,3)
# conv_layer.weight.data=kernel.data
# output=conv_layer(input)
# print(output)
#--------------------------------------------------------------------------
# import torch
# input=[3,4,6,5,7,\
#        2,4,6,8,2,\
#        1,6,7,8,4,\
#        9,7,4,6,2,\
#        3,7,5,4,1,\
#       ]
# input=torch.tensor(input,dtype=torch.float32).reshape(1,5,1,5)
# print(input.shape[0])
# print(input.size(0))
# --------------------------------------------------------------------
# import torch
# import numpy as np
# w=np.zeros(20)
# print(w)
# max_degree = 20 # 多项式的最⼤阶数
# n_train, n_test = 100, 100 # 训练和测试数据集⼤⼩
# true_w = np.zeros(max_degree) # 分配⼤量的空间
# true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])
# features = np.random.normal(size=(n_train + n_test, 1))
# print(features)
# np.random.shuffle(features)
# poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
# print(poly_features)
# print(poly_features.shape)
# a=torch.tensor([[1,1,2,3],[13,455,661,21]])
# print(a[:,1])
#----------------------------
# import torch
# import numpy as np
# import os
# x=np.array([[1,2,3,4,5],[1,23,4,56,2]])
# from PIL import Image
#----------------------------------------------
# import cv2
# img = cv2.imread("C:\\Users\\liu\\Desktop\\1\\CAZ~_(V}M5$)C3L9VQ$G$VD.jpg")
# cv2.imshow("I", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#------------------------------------
# import torch
# import os
# from PIL import Image#该包用来读取图片
# from torch.utils.data import Dataset
# class my_data(Dataset):
#     def __init__(self,dir1,dir2):
#        self.dir1=dir1
#        self.dir2=dir2
#        self.img_path=os.path.join(self.dir1,self.dir2)
#        self.img_list=os.listdir(self.img_path)
#     def __len__(self):
#         return len(self.img_list)
#     def __getitem__(self,i):
#         img_name = self.img_list[i]
#         img_=os.path.join(self.dir1,self.dir2,img_name)#图片路径
#         img=Image.open(img_)
#         label=self.dir2
#         return img,label
# x=my_data('D:\\download\\hymenoptera_data\\train','ants')
# y=my_data('D:\\download\\hymenoptera_data\\train','bees')
# img1,label1=x[0]
# img2,label2=y[0]
# img1.show()
# img2.show()
#-----------------------------------------------------
#     # def __init__(self,img_dir,label_dir):
#     #     self.img=img_dir
#     #     self.label=label_dir
#     #     self.path=os.path.join(self.img,self.label)
#     #     self.path=os.listdir(self.path)#将该文件夹下的文件变为列表以便得到对应的图片名字为获得图片信息做准备
#     # def __getitem__(self, idx):#返回对应图片信息和标签
#     #     img_name=self.path[idx]
#     #     img_item_path=os.path.join(self.img,self.label,img_name)
#     #     img=Image.open(img_item_path)
#     #     label=self.label
#     #     return img,label
#     # def __len__(self):
#     #     return len(self.path)
# # test=my_data('D:\\download\\hymenoptera_data\\train','ants')
# # img,lable=test[0]
# # img.show()
# # print(test[0])
#---------------------------------------------------------------
# a=[1,123,23,23,2,2]
# b=[213,123,1,2,2]
# y=a+b
# print(y)
#------------------------------------------------------
# from torchvision import transforms
# import torch
# from PIL import Image
# img_path="D:/download/hymenoptera_data/train/ants/0013035.jpg"
# img=Image.open(img_path)
# transform=transforms.ToTensor()
# x=transform(img)
# print(type(x))
# aaa=torch.tensor([1,1,233,3])
# print(type(aaa))
#-----------------------------------------------------
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
#----------------------------------------------------------------
# from torchvision import transforms
# import torch
# from PIL import Image
# img=Image.open("D:/download/hymenoptera_data/train/ants/0013035.jpg")
# img.show()
# x=transforms.Resize((200,200))#Resize是一个类需要实例化
# img=x(img)#x的返回值为PIL类型的
# img.show()
#------------------------------------
# from torchvision import transforms
# import torch
# from PIL import Image
# img=Image.open("D:/download/hymenoptera_data/train/ants/0013035.jpg")
# img.show()
# print(img)
# x=transforms.Resize((200,250))#Resize是一个类需要实例化
# img=x(img)#x的返回值为PIL类型的
# print(img)
# y=transforms.Resize(500)
# img=y(img)
# print(img)
#--------------------------------------------------
# from PIL import Image
# import torchvision.transforms as transforms
# transform = transforms.Compose([
#     transforms.Resize((200, 200)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])
# image = Image.open("D:/download/hymenoptera_data/train/ants/0013035.jpg")
# transformed_image = transform(image)
#---------------------------------------------------------------