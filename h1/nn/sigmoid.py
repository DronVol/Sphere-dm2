from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module
import numpy as np
import math

class Sigmoid(Module):
	def __init__(self, *args):
		super(Sigmoid, self).__init__(*args)
		self.name = "Sigmoid"
		self.param = None
		self.need_target = False
		assert self.input_size == self.output_size

	def sigm(self, z):
		return 1. / (1. + np.exp(-z))

	def forward(self, *args):
		super(Sigmoid, self).forward(*args)
		self.output = self.sigm(self.input)
		return self.output
	
	def backward(self, *args, **kwargs):
		super(Sigmoid, self).backward(*args, **kwargs)
		return np.multiply(self.next_grad, self.grad_input)
		
	def update_grad_input(self, *args, **kwargs):
		self.grad_input = np.multiply(self.output, (1. - self.output))
		
	def update_parameters(self, next_grad, learning_rate):
		self.next_grad = next_grad
		
	def local_gradient(self, inputs, target=None, eps=1e-3, tol=1e-3):
		self.forward(inputs, target)
		grad_an = np.multiply(self.output, (1. - self.output))
		grad_num = np.zeros(shape=np.prod(inputs.shape))
		
		for j, x in enumerate(np.nditer(inputs, op_flags = ['readwrite'])):
			x -= eps
			left = self.forward(inputs, target)
			x += 2 * eps
			right = self.forward(inputs, target)
			x -= eps
			der = (right - left) / (2. * eps)
			grad_num[j] = der.ravel()[j]
		grad_num = np.reshape(np.array(grad_num), newshape=grad_an.shape)
		norm = np.linalg.norm(grad_an - grad_num) / np.prod(grad_an.shape)
		print ("||Grad_input_num - Grad_input_an|| / Size = %.6f" % (norm))
		
		if norm < tol:
			return True
		else:
			return False