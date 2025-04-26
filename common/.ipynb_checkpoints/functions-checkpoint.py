import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(c-x)
    sum_exp_x = np.sum(np.exp(c-x))

    return exp_x / sum_exp_x