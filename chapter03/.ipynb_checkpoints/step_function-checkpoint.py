import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return np.array(x > 0, dtype=np.int8)

x = np.arange(-5, 5, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.show()