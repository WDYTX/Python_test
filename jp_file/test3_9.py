import numpy as np

# 定义被积函数
def f(x):
    # 使用np.maximum将小于等于0的值替换为1e-10以避免除以零
    x = np.maximum(x, 1e-10)
    return np.sqrt(x) * np.log(x)

# 复合梯形法
def composite_trapezoidal(a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    result = h * (0.5 * y[0] + 0.5 * y[-1] + np.sum(y[1:-1]))
    return result

# 复合辛普森法
def composite_simpson(a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    result = h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])
    return result

# 精确值
exact_value = -4/9

# 不同的步长
step_sizes =[10**i for i in range(8)]

# 计算误差
errors_trapezoidal = []
errors_simpson = []

for n in step_sizes:
    estimated_value_trapezoidal = composite_trapezoidal(0, 1, n)
    estimated_value_simpson = composite_simpson(0, 1, n)
    errors_trapezoidal.append(abs(exact_value - estimated_value_trapezoidal))
    errors_simpson.append(abs(exact_value - estimated_value_simpson))

# 绘制误差图
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(step_sizes, errors_trapezoidal, label='Composite Trapezoidal', marker='o')
plt.plot(step_sizes, errors_simpson, label='Composite Simpson', marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Step Size (log scale)')
plt.ylabel('Error (log scale)')
plt.legend()
plt.title('Error vs. Step Size for Numerical Integration')
plt.grid()
plt.show()
