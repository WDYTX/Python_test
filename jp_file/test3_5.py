import numpy as np
import matplotlib.pyplot as plt

# 定义龙格函数
def f(x):
    return 1 / (1 + 40 * x**2)

# 定义分段线性插值函数
def linear_interpolation(x, x_values, y_values):
    n = len(x_values)
    for i in range(n - 1):
        if x_values[i] <= x <= x_values[i + 1]:
            return (x - x_values[i]) * (y_values[i + 1] - y_values[i]) / (x_values[i + 1] - x_values[i]) + y_values[i]
    return 0  # Handle x values outside the range

# 定义要插值的区间和节点数量
n1 = 11
n2 = 21
x_values1 = np.linspace(-5, 5, n1)
x_values2 = np.linspace(-5, 5, n2)
y_values1 = f(x_values1)
y_values2 = f(x_values2)

# 生成插值点
x_interp = np.linspace(-5, 5, 100000)
y_interp1 = [linear_interpolation(x, x_values1, y_values1) for x in x_interp]
y_interp2 = [linear_interpolation(x, x_values2, y_values2) for x in x_interp]

# 绘制原始函数和插值结果
plt.figure(figsize=(12, 6))
plt.plot(x_interp, f(x_interp), label='f(x)', linewidth=2)
plt.plot(x_interp, y_interp1, label=f'Interpolation (n={10})')
plt.plot(x_interp, y_interp2, label=f'Interpolation (n={20})')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.title('Linear Interpolation')
plt.show()
