# #!/usr/bin/env python
# # coding: utf-8
#
# # In[1]:
#
#
# messenge="hehe123"
# print(messenge)
#
#
# # In[2]:
#
#
# messenge="123354"
# print(messenge)
# messenge="nIbiplus sfAs asDccddz 123"
# print(messenge.title())
# print(messenge.upper())
# print(messenge.lower())
#
#
# # In[3]:
#
#
# a="7ddd"
# b="sjd"
# c=a+b
# print(c)
#
#
# # In[4]:
#
#
# print('python3')
# print('\tpython3')
# print('python3\t')
# print('python3\n')
#
#
# # ### hhhh
# # # hhhh
# # ## hhhh
# # #### hhhh
#
# # In[9]:
#
#
# messenge='languages:\n\tpython\n\tc\n\tjava_script'
# print(messenge)
#
#
# # In[18]:
#
#
# a=' ssd  '
# b=a.rstrip()
# print(b)
# b=b.lstrip()
# print(b)
# print('123')
#
#
# # In[21]:
#
#
# a=' yxz '
# b=a.strip()
# print(b)
#
#
# # In[23]:
#
#
# x='sdksd'
# y='s_jdoi'
# print("sdfdsfg'xxx'")
import torch
from IPython import display
from d2l import torch as d2l
a=torch.tensor([[1,2,3],[4,5,6]])
b=a.sum(1,keepdim=True)
print(a,b)
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
print(y_hat[[0, 1], y])





