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
print(torch.arange(12))
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    print(X)
    print('\n')
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    print(y)
    return X, y.reshape((-1, 1))
true_w = torch.tensor([2,-3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 10000)
for i in range(10000):
    print('features:', features[0],'\nlabel:', labels[0])



