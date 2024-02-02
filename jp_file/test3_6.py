import numpy as np
import matplotlib.pyplot as plt


# 定义龙格函数
def f(x):
    return 1 / (1 + 40 * x ** 2)


# 定义三次样条插值函数
def cubic_spline_interpolation(x, x_values, y_values):
    n = len(x_values)
    h = np.diff(x_values)
    alpha = np.zeros(n)
    beta = np.zeros(n)

    for i in range(1, n - 1):
        alpha[i] = 3 * (y_values[i + 1] - 2 * y_values[i] + y_values[i - 1]) / (h[i] * h[i])
        beta[i] = alpha[i - 1] + alpha[i]

    c = np.zeros(n)
    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)

    for i in range(1, n - 1):
        l[i] = 2 * (x_values[i + 1] - x_values[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    c[-1] = 0
    l[-1] = 1
    z[-1] = 0

    b = np.zeros(n)
    d = np.zeros(n)

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y_values[j + 1] - y_values[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    # 插值
    result = np.zeros_like(x)
    for i in range(n - 1):
        mask = (x >= x_values[i]) & (x <= x_values[i + 1])
        result[mask] = (y_values[i] +
                        (x[mask] - x_values[i]) * (b[i] +
                                                   (x[mask] - x_values[i]) * (c[i] +
                                                                              (x[mask] - x_values[i]) * d[i])))

    return result


# 定义要插值的区间和节点数量
n1 = 11
n2 = 21
x_values1 = np.linspace(-5, 5, n1)
x_values2 = np.linspace(-5, 5, n2)
y_values1 = f(x_values1)
y_values2 = f(x_values2)

# 生成插值点
x_interp = np.linspace(-5, 5, 100000)
y_interp1 = cubic_spline_interpolation(x_interp, x_values1, y_values1)
y_interp2 = cubic_spline_interpolation(x_interp, x_values2, y_values2)

# 绘制原始函数和插值结果
plt.figure(figsize=(12, 6))
plt.plot(x_interp, f(x_interp), label='f(x)', linewidth=2)
# plt.plot(x_interp, y_interp1, label=f'Cubic Spline Interpolation (n={10})')
plt.plot(x_interp, y_interp2, label=f'Cubic Spline Interpolation (n={20})')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.title('Cubic Spline Interpolation')
plt.show()
