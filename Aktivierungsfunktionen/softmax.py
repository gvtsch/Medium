import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Consolas"

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def softmax_derivative(x):
    s = softmax(x)
    return s * (1 - s)

x = np.linspace(-10, 10, 100)

fig, ax = plt.subplots()
ax.plot(x, softmax(x), label="softmax(x)")
ax.plot(x, softmax_derivative(x), label="softmax'(x)")
ax.grid(True)
ax.legend()

plt.show()