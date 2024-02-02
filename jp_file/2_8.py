import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 元学习模型定义
class MetaLearner(nn.Module):
    def __init__(self):
        super(MetaLearner, self).__init__()
        self.model = nn.Linear(1, 1)  # 简单的线性模型

    def forward(self, x):
        return self.model(x)

# MAML算法
def maml(model, x, y, lr_inner=0.01, num_iterations=1):
    model_copy = MetaLearner()
    model_copy.load_state_dict(model.state_dict())
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model_copy.parameters(), lr=lr_inner)

    for _ in range(num_iterations):
        predictions = model_copy(x)
        loss = criterion(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model_copy

# 数据生成
def generate_task():
    x = np.random.rand(10, 1)  # 任务数据
    y = 3 * x + 2 + np.random.randn(10, 1) * 0.1  # 对应的标签，加上噪声
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# 主元学习过程
meta_model = MetaLearner()
meta_optimizer = optim.SGD(meta_model.parameters(), lr=0.001)

for meta_iteration in range(1000):
    task_x, task_y = generate_task()

    # 在任务上进行元训练
    model_prime = maml(meta_model, task_x, task_y, lr_inner=0.01, num_iterations=5)

    # 在任务上进行元测试
    test_x, test_y = generate_task()
    predictions = model_prime(test_x)
    meta_loss = nn.MSELoss()(predictions, test_y)

    # 元优化
    meta_optimizer.zero_grad()
    meta_loss.backward()
    meta_optimizer.step()

    if meta_iteration % 100 == 0:
        print(f'Meta Iteration {meta_iteration}, Meta Loss: {meta_loss.item()}')

# 使用元学习后的模型进行预测
new_task_x, new_task_y = generate_task()
new_predictions = meta_model(new_task_x)
print('Predictions on new task:', new_predictions.detach().numpy())
