import numpy as np


# Define the function to be integrated
def f(x):
    if x <= 0:
        return 0  # or any other suitable value
    else:
        return np.sqrt(x) * np.log(x)



# Exact integral value
exact_value = -4 / 9

# Integration limits
a = 0
b = 1

# Range of step sizes (h values)
h_values = np.logspace(-8, 0, 7)

# Initialize lists to store errors
trapezoidal_errors = []
simpson_errors = []

for h in h_values:
    # Number of subintervals
    n = int((b - a) / h)

    # Trapezoidal rule
    trapezoidal_integral = 0
    for i in range(1, n):
        x_i = a + i * h
        trapezoidal_integral += f(x_i)
    trapezoidal_integral += (f(a) + f(b)) / 2
    trapezoidal_integral *= h

    trapezoidal_error = abs(exact_value - trapezoidal_integral)
    trapezoidal_errors.append(trapezoidal_error)

    # Simpson's rule
    simpson_integral = 0
    for i in range(1, n, 2):
        x_i = a + i * h
        simpson_integral += 4 * f(x_i)
    for i in range(2, n - 1, 2):
        x_i = a + i * h
        simpson_integral += 2 * f(x_i)
    simpson_integral += f(a) + f(b)
    simpson_integral *= h / 3

    simpson_error = abs(exact_value - simpson_integral)
    simpson_errors.append(simpson_error)

# Plot the errors as a function of h
import matplotlib.pyplot as plt

plt.loglog(h_values, trapezoidal_errors, label='Trapezoidal Error')
plt.loglog(h_values, simpson_errors, label="Simpson's Error")
plt.xlabel('Step Size (h)')
plt.ylabel('Error')
plt.legend()
plt.show()




