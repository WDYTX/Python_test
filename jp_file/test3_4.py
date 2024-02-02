
import numpy as np
import matplotlib.pyplot as plt

# 定义要进行插值的函数
def f(x):
    return 1 / (1 + 40 * x**2)

# 定义牛顿插值函数
def newton_interpolation(x, y, xi):
    n = len(x)
    coefficients = y.copy()

    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            coefficients[j] = (coefficients[j] - coefficients[j - 1]) / (x[j] - x[j - i])

    result = coefficients[-1]
    for i in range(n - 2, -1, -1):
        result = result * (xi - x[i]) + coefficients[i]

    return result

# 创建等距节点
n1 = 11
n2 = 21
x1 = np.linspace(-5, 5, n1)
x2 = np.linspace(-5, 5, n2)
y1 = f(x1)
y2 = f(x2)

# 生成插值函数的值
xi = np.linspace(-5, 5, 100000)
yi1 = [newton_interpolation(x1, y1, xi_val) for xi_val in xi]
yi2 = [newton_interpolation(x2, y2, xi_val) for xi_val in xi]

# 绘制图像
plt.figure(figsize=(12, 6))
plt.plot(xi, f(xi), label='f(x)', linewidth=2)
plt.plot(xi, yi1, label=f'Newton Interpolation (n={10})')
plt.plot(xi, yi2, label=f'Newton Interpolation (n={20})')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Newton Interpolation ')
plt.legend()
plt.grid()
plt.show()
#--------------------------------------------












