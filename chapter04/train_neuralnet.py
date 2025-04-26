import os
import sys
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
	batch_mask = np.random.choice(train_size, batch_size)
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]

	grad = network.gradient(x_batch, t_batch)

	# 更新参数
	for key in ('W1', 'b1', 'W2', 'b2'):
		network.param[key] -= learning_rate * grad[key]

	loss = network.loss(x_batch, t_batch)
	train_loss_list.append(loss)

	if i % iter_per_epoch == 0:
		train_acc = network.accuracy(x_train, t_train)
		test_acc = network.accuracy(x_train, t_train)
		train_acc_list.append(train_acc)
		test_acc_list.append(test_acc)
		print(f"train acc = {train_acc}, test acc = {test_acc}")

figure, ax = plt.subplots(1, 3)
ax[0].plot(np.arange(len(train_acc_list)), train_acc_list, label="train acc", marker='o')
ax[0].set_xlabel = "epoch"
ax[0].set_ylabel = "accuracy"
ax[0].set_ylim(0, 1.)
ax[0].legend(loc="lower right")
ax[0].set_title("Train Accuracy")

ax[1].plot(np.arange(len(test_acc_list)), test_acc_list, label="test acc", linestyle="--", marker='s')
ax[1].set_xlabel = "epoch"
ax[1].set_ylabel = "accuracy"
ax[1].set_ylim(0, 1.)
ax[1].legend(loc="lower right")
ax[1].set_title("Test Accuracy")

ax[2].plot(np.arange(len(train_loss_list)), train_loss_list, label="train loss")
ax[2].set_xlabel = "epoch"
ax[2].set_ylabel = "loss"
ax[2].legend(loc="upper right")
ax[2].set_title("Train loss")

# plt.tight_layout()
plt.show()
