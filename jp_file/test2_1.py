# import torch
# from IPython import display
# from d2l import torch as d2l
# batch_size = 256
# # train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# a = torch.tensor([[1.,2.,3.,4.],[1.,2.,3.,4.]])
# print(a.mean())
# class a:
#     _x=10
#     def __init__(self):#构造方法
#         print("123")
# b=a()
# print(b._x)
# class Person:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age
#
#     def introduce(self):
#         print(f"My name is {self.name} and I am {self.age} years old.")
#
# # 创建Person类的实例
# person1 = Person("Alice", 25)
# person1.introduce()
import numpy as np
import matplotlib. pyplot as plt
# x_data =[1.0,2.0,3.0]
# y_data = [2.0,4.0,6.0]
# def forward(x):
#     return x*w
# for w in np.arange(1,4):
#     print(forward(x_data))
# import torch
# a=torch.tensor([[1,2,3,4],[32,412,4123,41]])
# b=a+1
# print(b)
# def f(*arg):
#     for i in range(len(arg)):
#         print(arg[i])
# f(1,2,45,566,74,'qw',1)
#-----------------------------------------------------
# import torch
# a=torch.tensor([[1.,2.,3.,4.],[123.,232.,12.,3.]],requires_grad=True)
# a.requires_grad=True
# import torch
# x_data=[1.0,2.0,3.0]
# y_data=[2.0,4.0,6.0]
# w=torch.tensor([1.0],requires_grad=True)
# def loss(y_hat,y):
#     return (y_hat-y)**2
# def forward(x):
#     return w*x
# for i in range(100):
#     for x,y in zip(x_data,y_data):
#         y_hat=forward(x)
#         l=loss(y_hat,y)
#         l.backward()#求梯度
#         w.data=w.data-0.1*w.grad.data
#         w.grad.data.zero_()
#     print('第{}次'.format(i))
#     print('损失为',l.item())
# print(w)
#------------------------------------------------------------------------------




# import torch
# def data_set():
#     yield  torch.tensor([1.0,2.0,3.0],requires_grad=True), torch.tensor([2.0,4.0,6.0])
# w=torch.tensor([1.0],requires_grad=True)
# def loss(y_hat, y):
#     return (y_hat-y)**2/len(y)
# def forward(x):
#     return w*x
# for i in range(100):
#     for x,y in data_set():
#         y_hat=forward(x)
#         l=loss(y_hat,y)
#         l.sum().backward()
#         w.data-=0.1*w.grad.data
#         w.grad.data.zero_()
#         sunshi = loss(y_hat, y)
#     print('迭代{}次,损失为{}'.format(i+1,sunshi.mean()))
#     print(w.data)
# for x,y in data_set():
#     print(x)
#     print(y)

#-----------------------------------------------------------------------------

# import torch
# def data_set():
#     yield  torch.tensor([1.0,2.0,3.0],requires_grad=True), torch.tensor([2.0,4.0,6.0])
# w=torch.tensor([1.0], requires_grad=True)
# def loss(y_hat,y):
#     return (y_hat-y)**2/len(y)
# def forward(x):
#     return w*x
# for i in range(100):
#     for x,y in data_set():
#         y_hat=forward(x)
#         l=loss(y_hat,y)
#         l.sum().backward()#求梯度
#         w.data=w.data-0.1*w.grad.data
#         w.grad.data.zero_()
#     print('第{}次'.format(i))
#     print('损失为',l.sum().item())
# print(w.item())
#--------------------------------------------------------------------------
# import torch
# a=torch.tensor([[1,2],[5,6],[3,4],[12,44]])
# w=3
# b=w*a
# print(b)
# def fff():
#     yield a[b]
# for c in fff():
#     print(c)
# print(a[b])
# class ppp():
#
#     def __call__(self,i):
#         print(i)
# p=ppp()
# p(1)

#-------------------------------------------------------------------------------
# import torch
# x_data = torch.Tensor([[1.0], [2.0], [3.0]])
# y_data = torch.Tensor([[2.0], [4.0], [6.0]])
# class LinearModel(torch.nn.Module):
#     def __init__(self):
#         super(LinearModel, self).__init__()
#         self.linear = torch.nn.Linear(1, 1)
#         # 包含了权重和偏置的计算  linear 也是继承 Module  1，1 是每一个输入和输出样本
#     def forward(self, x):
#         y_pred = self.linear(x)
#         return y_pred
# model = LinearModel()# 实例化
# criterion = torch.nn.MSELoss(reduction='sum')# 是否求均值
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# for epoch in range(1000):
#     y_pred = model(x_data)
#     loss = criterion(y_pred, y_data)
#     print(epoch, loss.item())
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
# print('w = ', model.linear.weight.item())
# print('b = ', model.linear.bias.item())
# x_test = torch.Tensor([[4.0]])
# y_test = model(x_test)
# print('y_pred = ', y_test.data)
#-------------------------------------------------------------------------------
# import torch
# x=torch.tensor([[1.0],[2.0],[3.0]])
# y=torch.tensor([[2.0],[4.0],[6.0]])
# class Linear(torch.nn.Module):#创建一个继承module的类
#     def __init__(self):
#         super(Linear,self).__init__()#使用父类的构造函数
#         self.llinear=torch.nn.Linear(1,1)#
#     def forward(self ,x):
#         y_hat=self.llinear(x)
#         return y_hat
# lllinear=Linear()#类实例化
# #均方损失
# loss=torch.nn.MSELoss(reduction='sum')
# #更新
# updata=torch.optim.SGD(lllinear.parameters(),lr=0.01)
# #训练
# for epoch in range(100):
#     y_hat=lllinear(x)
#     print(y_hat)
#     print('\n')
#     l=loss(y_hat,y)#求损失
#     print(epoch,l.item())
#     updata.zero_grad()#清零梯度
#     l.backward()
#     updata.step()
# print('w = ',lllinear.llinear.weight.item())
# print('b = ',lllinear.llinear.bias.item())
# x_test = torch.Tensor([[4.0]])
# y_test = lllinear(x_test)
# print('y_pred = ', y_test.data)
#---------------------------------------------------------------------
# import torch
# x_data = torch.tensor([[1.0], [2.0], [3.0]])
# y_data=torch.tensor([[2.0],[4.0],[6.0]])
# class Linearmodule(torch.nn.Module):#定义模型并继承module
#     def __init__(self):
#         super(Linearmodule, self).__init__()
#         self.linearlayer=torch.nn.Linear(1,1)
#     def forward(self,x_data):
#         return self.linearlayer(x_data)
# #类的实例化
# linearmodule=Linearmodule()
# #损失
# loss_f=torch.nn.MSELoss()
# #更新器
# updata_f=torch.optim.SGD(linearmodule.parameters(),lr=0.1)
# #训练
# for i in range(1000):
#     y_hat=linearmodule(x_data)
#     l=loss_f(y_hat,y_data)
#     updata_f.zero_grad()
#     l.backward()
#     updata_f.step()
#     ttt=loss_f(linearmodule(x_data),y_data)
#     print("第{}次迭代，其损失为{:.100f}".format(i+1,ttt))
#     if ttt==0.0:
#         break
# print("迭代后权重w为：",linearmodule.linearlayer.weight.item())
# print('总共迭代{}次'.format(i+1))
#-----------------------------------------------------------------
# import torch
# a=torch.tensor([[1,2,3,4],[3,4,5,6]])
# b=torch.tensor([[2,4,6,8],[4,23,888,655]])
# for z,y in zip(a,b):
#     print(z,y)
#----------------------------------------------------------------------
# import torch
# #准备数据
# x_data = torch.tensor([[1.0], [2.0], [3.0]])
# y_data=torch.tensor([[0.],[0.],[1.]])
# #定义类---用类来实现模型
# class SF(torch.nn.Module):
#     def __init__(self):
#         super(SF, self).__init__()
#         self.sf1=torch.nn.Linear(1,1)
#     def forward(self,x_data):
#         return torch.nn.functional.sigmoid(self.sf1(x_data))
# #类的实例化
# sf=SF()
# #损失
# loss=torch.nn.BCELoss()
# #更新器
# updata=torch.optim.SGD(sf.parameters(),lr=0.01)
# #开始迭代/训练
# for i in range(1000):
#     #用模型算出预测值
#     y_hat=sf(x_data)
#     #求损失
#     l=loss(y_hat,y_data)
#     #梯度清零
#     updata.zero_grad()
#     #求梯度
#     l.backward()
#     #更新
#     updata.step()
# import numpy as np
# import matplotlib.pyplot as plt
# x = np.linspace(0, 10, 200)
# x_t = torch.Tensor(x).view((200, 1))
# y_t = sf(x_t)
# y = y_t.data.numpy()
# plt.plot(x, y)
# plt.plot([0, 10], [0.5, 0.5], c='r')
# plt.xlabel('Hours')
# plt.ylabel('Probability of Pass')
# plt.grid()
# plt.show()
# b=sf(torch.tensor([2.6]))
# print(b.item())
#----------------------------------------------------------------------------
# import numpy as np
# import torch
#
# numpy_array = np.array([1,2,32,4,5])  # 输入你的NumPy数组
# tensor = torch.from_numpy(numpy_array)
# print(tensor)
#----------------------------------------------------------------------------



















