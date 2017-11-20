import numpy as np

class FirstOrder:
	
	def __call__(self,function,x):
		gradient = function.grad_f(x)
		return gradient 
	
	def update_state(self, function, x, old_x):
	    pass
		
class SecondOrder:
	
	def __call__(self,function,x):
		grad_2 = function.grad2_f(x)
		grad_1 = function.grad_f(x)
		grad_inv = np.linalg.inv(grad_2)
		update = np.matmul(grad_inv,grad_1)
		return update
	
	def update_state(self, function, x, old_x):
	    pass
	
class BFGS:
	
	def __init__(self,initial_H):
		self.H = initial_H
		return
		
	def __call__(self,function, x):
	    return np.matmul(self.H, function.grad_f(x))
	    
	def update_state(self, function, x, old_x):
	    old_grad = function.grad_f(old_x)
	    grad = function.grad_f(x)
	    s = x - old_x
	    y = grad - old_grad
	    I = np.diag([1]*len(y))
	    lhs = I - np.outer(s, y)/np.dot(y, s) 
	    rhs = I - np.outer(y, s)/np.dot(y, s)
	    a = np.matmul(lhs, self.H)
	    a = np.matmul(a, rhs)
	    self.H = a + np.outer(s, s)/np.dot(y, s)
	    return
