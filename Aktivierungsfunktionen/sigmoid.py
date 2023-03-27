import matplotlib.pyplot as plt 
import numpy as np 
plt.rcParams["font.family"] = "Consolas"

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

x = np.linspace(-10, 10, 1000) 
fig, ax = plt.subplots() 
ax.plot(x, sigmoid(x), label="sigmoid(x)")
ax.plot(x, sigmoid_derivative(x), label="sigmoid'(x)")
ax.grid(True)
ax.legend()

plt.show()