import os
import sys

sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.optimizer import SGD
from common.multi_layer_net_extend import MultiLayerNetExtend

# load dataset
(x_train, t_train), (x_test, t_test) = load_mnist()
x_train = x_train[:300]
t_train = t_train[:300]

network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100] * 6, output_size=10,
                              use_dropout=True, dropout_ratio=0.2)
train_size = x_train.shape[0]
batch_size = 100
max_epochs = 201
learning_rate = 0.01
optimizer = SGD(lr=learning_rate)
iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

train_acc_list = []
test_acc_list = []

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(f"epoch: {epoch_cnt}, train acc: {train_acc}, test acc: {test_acc}")

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break

# draw
markers = {"train": "o", "test": "s"}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker=markers["train"], label="train_acc", markevery=10)
plt.plot(x, test_acc_list, marker=markers["test"], linestyle='--', label="test_acc", markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1)
plt.legend(loc="best")
plt.show()