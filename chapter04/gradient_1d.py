import numpy as np
import matplotlib.pyplot as plt

# 0.01x^2 + 0.1x
def function_1(x):
	return 0.01 * x ** 2 + 0.1 * x

# 中心差分近似导数, 也就是斜率
def numerical_diff(f, x):
	h = 1e-4
	return (f(x+h) - f(x-h)) / (2 * h)

# 计算切线方程
def target_line(f, x):
	d = numerical_diff(f, x)
	print(d)
	# y计算的是截距 y=d*x+b 带入(x, f(x))
	y = f(x) - d*x
	return lambda t: d*t + y


x = np.arange(0., 20., 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = target_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()

