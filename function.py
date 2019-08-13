Python 3.7.1 (v3.7.1:260ec2c36a, Oct 20 2018, 14:57:15) [MSC v.1915 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import numpy as np
>>> def numerical_gradient(f,x):
	h = 1e-4
	grad = np.zero_like(x)
	for idx in range(x.size):
		tmp_val = x[idx]
		x[idx] = tmp_val + h
		fxh1 = f(x)
		x[idx] = tmp_val - h
		fxh2 = f(x)
		grad[idx] = (fxh1 - fxh2)/(2*h)
		x[idx] = tmp_val
	return grad

>>> def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
	x = init_x
	for i in range(step_num):
		grad = numerical_gradient(f, x)
		x = lr * grad
	return x

>>> def function_2(x):
	return x[0]**2 + x[1]**2

>>> init_x = np.array([-3.0, 4.0])
>>> gradient_descent(function_2, init_x = init_x, lr = 0.0, step_num = 100)
Traceback (most recent call last):
  File "<pyshell#25>", line 1, in <module>
    gradient_descent(function_2, init_x = init_x, lr = 0.0, step_num = 100)
  File "<pyshell#20>", line 4, in gradient_descent
    grad = numerical_gradient(f, x)
  File "<pyshell#13>", line 3, in numerical_gradient
    grad = np.zero_like(x)
AttributeError: module 'numpy' has no attribute 'zero_like'
>>> def numerical_gradient(f,x):
	h = 1e-4
	grad = np.zeros_like(x)
	for idx in range(x.size):
		tmp_val = x[idx]
		x[idx] = tmp_val + h
		fxh1 = f(x)
		x[idx] = tmp_val - h
		fxh2 = f(x)
		grad[idx] = (fxh1 - fxh2)/(2*h)
		x[idx] = tmp_val
	return grad

>>> def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
	x = init_x
	for i in range(step_num):
		grad = numerical_gradient(f, x)
		x = lr * grad
	return x

>>> def function_2(x):
	return x[0]**2 + x[1]**2

>>> init_x = np.array([-3.0, 4.0])
>>> gradient_descent(function_2, init_x = init_x, lr = 0.0, step_num = 100)
array([0., 0.])
>>> def softmax(a):
	c = np.max(a)
	exp_a = np.exp(a-c)
	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a
	return y

>>> def cross_entropy_error(y, t):
	if y.ndim == 1:
		t = t.reshape(1, t.size)
		y = y.reshape(1,y.size)
	batch_size = y.shape[0]
	return -np.sum(np.log(y[np.arange(batch_size),t]+ 1e-7))/batch_size

>>> 
>>> import sys,os
>>> sys.path.append(os.pardir)
>>> from common.functions import softmax,cross_entropy_error
Traceback (most recent call last):
  File "<pyshell#51>", line 1, in <module>
    from common.functions import softmax,cross_entropy_error
ModuleNotFoundError: No module named 'common'
>>> from common.gradient import numerical_gradient
Traceback (most recent call last):
  File "<pyshell#52>", line 1, in <module>
    from common.gradient import numerical_gradient
ModuleNotFoundError: No module named 'common'
>>> 
>>> class simpleNet:
	def__init__(self):
		
SyntaxError: invalid syntax
>>> class simpleNet:
	def __init__(self):
		self.W = np.random.randn.(2,3)
		
SyntaxError: invalid syntax
>>> class simpleNet:
	def __init__(self):
		self.W = np.random.randn(2,3)
	def predict(self,x):
		return np.dot(x, self.W)
	def loss(self, x, t):
		z = self.predict(x)
		y = softmax(z)
		loss = cross_entropy_error(y,t)
		return

	
>>> net = simpleNet()
>>> print(net.W)
[[0.43308984 0.24684501 0.08187988]
 [2.17885846 0.99888234 1.69304333]]
>>> print(net.W)
[[0.43308984 0.24684501 0.08187988]
 [2.17885846 0.99888234 1.69304333]]
>>> 
>>> x = np.array([0.6,0.9])
>>> p = net.predict(x)
>>> print(p)
[2.22082652 1.04710111 1.57286693]
>>> np.argmax(p)
0
>>> t = np.arrar([0,0,1])
Traceback (most recent call last):
  File "<pyshell#75>", line 1, in <module>
    t = np.arrar([0,0,1])
AttributeError: module 'numpy' has no attribute 'arrar'
>>> t = np.array([0,0,1])
>>> net.loss(x,t)
>>> 
>>> net.loss(x,t)
>>> 
