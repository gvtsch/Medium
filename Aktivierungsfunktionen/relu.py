import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = "Consolas"

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return 1. * (x > 0)


x = np.linspace(-10, 10, 100)

fig, ax = plt.subplots()
ax.plot(x, relu(x), label="relu(x)")
ax.plot(x, relu_derivative(x), label="relu'(x)")
ax.grid(True)
ax.legend()

plt.show()