from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module
import numpy as np

class MSE(Module):
    def __init__(self, *args):
        super(MSE, self).__init__(*args)
        self.name = "MSE"
        self.need_target = True
        self.param = None
    
    def forward(self, *args, **kwargs):
        super(MSE, self).forward(*args, **kwargs)
        self.answers = self.input
        self.output = ((self.input - self.target) ** 2).sum().sum() / self.input.shape[0]
        return self.output
    
    def backward(self, *args, **kwargs):
    	super(MSE, self).backward(*args, **kwargs)
        return self.grad_input
    	
    def update_grad_input(self, *args, **kwargs):
    	self.grad_input = 2 * (self.input - self.target) / self.input.shape[0]
        return self.grad_input

    def update_parameters(self, next_grad, learning_rate):
        self.next_grad = next_grad
        pass

    def gradient_check_local(self, *args):
        pass