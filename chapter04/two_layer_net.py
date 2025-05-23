import os, sys
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, sigmoid, cross_entropy_error, sigmoid_grad
from common.gradient import numerical_gradient

class TwoLayerNet():
	def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.1):
		self.param = {}
		self.param['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
		self.param['b1'] = np.zeros(hidden_size)
		self.param['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
		self.param['b2'] = np.zeros(output_size)

	def predict(self, x):
		W1, W2 = self.param['W1'], self.param['W2']
		b1, b2 = self.param['b1'], self.param['b2']

		a1 = np.dot(x, W1) + b1
		z1 = sigmoid(a1)
		a2 = np.dot(z1, W2) + b2
		y = softmax(a2)

		return y

	def loss(self, x, t):
		y = self.predict(x)

		return cross_entropy_error(y, t)


	def accuracy(self, x, t):
		y = self.predict(x)
		y = np.argmax(y, axis=1)
		t = np.argmax(t, axis=1)
		accuracy = np.sum(y == t) / float(x.shape[0])

		return accuracy

	def numerical_gradient(self, x, t):
		loss_W = lambda W: self.loss(x, t)

		grads = {}
		grads['W1'] = numerical_gradient(loss_W, self.param['W1'])
		grads['b1'] = numerical_gradient(loss_W, self.param['b1'])
		grads['W2'] = numerical_gradient(loss_W, self.param['W2'])
		grads['b2'] = numerical_gradient(loss_W, self.param['b2'])

		return grads

	def gradient(self, x, t):
		W1, W2 = self.param['W1'], self.param['W2']
		b1, b2 = self.param['b1'], self.param['b2']

		grads = {}
		batch_num = x.shape[0]

		# forward
		a1 = np.dot(x, W1) + b1
		z1 = sigmoid(a1)
		a2 = np.dot(z1, W2) + b2
		y = softmax(a2)

		# backward
		dy = (y-t) / batch_num
		grads['W2'] = np.dot(z1.T, dy)
		grads['b2'] = np.sum(dy, axis=0)

		da1 = np.dot(dy, W2.T)
		dz1 = sigmoid_grad(a1) * da1
		grads['W1'] = np.dot(x.T, dz1)
		grads['b1'] = np.sum(dz1, axis=0)

		return grads



