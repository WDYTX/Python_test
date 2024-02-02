import numpy as np
import matplotlib.pyplot as plt
def cubic_spline_interpolation(x, x_nodes, y_nodes):
    n = len(x_nodes)
    h = np.diff(x_nodes)
    delta = np.diff(y_nodes) / h
    A = np.zeros((n, n))
    b = np.zeros_like(y_nodes)
    for i in range(1, n-1):
        A[i, i-1:i+2] = [h[i-1], 2 * (h[i-1] + h[i]), h[i]]
        b[i] = 3 * (delta[i] - delta[i-1])
    A[0, 0] = 1
    A[-1, -1] = 1
    c = np.linalg.solve(A, b)
    d = np.diff(c) / (3 * h)
    b = delta - h * (2 * c[:-1] + c[1:]) / 3
    x_idx = np.searchsorted(x_nodes, x)
    x_idx = np.clip(x_idx, 1, n-1) - 1
    dx = x - x_nodes[x_idx]
    y = y_nodes[x_idx] + b[x_idx] * dx + c[x_idx] * dx**2 + d[x_idx] * dx**3
    return y
def f(x, k):
    return 1 / (1 + k * x**2)
n1 = 11
n2 = 21
k = 40
x_nodes1 = np.linspace(-5, 5, n1)
x_nodes2 = np.linspace(-5, 5, n2)
y_nodes1 = f(x_nodes1, k)
y_nodes2 = f(x_nodes2, k)
x = np.linspace(-5, 5, 100000)
y_interp1 = cubic_spline_interpolation(x, x_nodes1, y_nodes1)
y_interp2 = cubic_spline_interpolation(x, x_nodes2, y_nodes2)
y_true = f(x, k)
plt.plot(x, y_true, label='f(x)')
plt.plot(x, y_interp1, label='Interpolation (n=10)')
plt.plot(x, y_interp2, label='Interpolation (n=20)')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cubic Spline Interpolation')
plt.show()