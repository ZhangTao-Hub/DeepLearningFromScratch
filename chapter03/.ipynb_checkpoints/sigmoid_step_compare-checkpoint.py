import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return np.array(x > 0, np.int8)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5, 5, 0.1)
y1 = step_function(x)
y2 = sigmoid(x)
plt.plot(x, y1)
plt.plot(x, y2, 'k--')
plt.show()

