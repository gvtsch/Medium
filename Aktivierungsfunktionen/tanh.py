import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Consolas"

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

x = np.linspace(-5, 5, 100)
fig, ax = plt.subplots()
ax.plot(x, tanh(x), label="tanh(x)")
ax.plot(x, tanh_derivative(x), label="tanh'(x)")
ax.grid(True)
ax.legend()

plt.show()