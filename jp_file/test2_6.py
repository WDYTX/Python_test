import test2_7
# print(test2_7.a)
import torch
import torch.nn as nn
from thop import profile

# 创建一个示例的神经网络模型
class SampleModel(nn.Module):
    def __init__(self):
        super(SampleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(262144, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# 创建一个示例的输入张量
input_tensor = torch.randn(1, 3, 64, 64)  # (batch_size, channels, height, width)

# 初始化神经网络模型
model = SampleModel()

# 使用Thop的`profile`函数来计算FLOP和参数数量
flops, params = profile(model, inputs=(input_tensor,))
print(f"FLOP count: {flops}")
print(f"Parameter count: {params}")
target=torch.randint(1,3,(1,80,80,80))
target[target == 4]=3
print(target)


