import numpy as np

class FirstOrder:
	
	def __call__(self,function,x):
		gradient = function.grad_f(x)
		return gradient 
	
	def update_state(self, x, old_x):
	    pass
		
class SecondOrder:
	
	def __call__(self,function,x):
		grad_2 = function.grad2_f(x)
		grad_1 = function.grad_f(x)
		grad_inv = np.linalg.inv(grad_2)
		update = np.matmul(grad_inv,grad_1)
		return update
	
	def update_state(self, x, old_x):
	    pass
	
class BFGS:
	
	def __init__(self,initial_H):
		self.H = initial_H
		return
		
	def __call__(self,function,x):
	    return np.matmul(self.H, function.grad_1(x))
	    
	def update_state(self, x, old_x):
	    old_grad = function.grad_1(old_x)
	    grad = function.grad_1(x)
	    s = x - old_x
	    y = old_grad - grad
	    I = np.diag([1]*len(y))
	    lhs = I - np.matmul(s, y.T)/np.dot(y, s) 
	    rhs = I - np.matmul(y, s.T)/np.dot(y, s)
	    a = np.matmul(lhs, self.H)
	    a = np.matmul(a, rhs)
	    self.H = a + np.matmul(s, s.T)/np.dot(y, s)
	    return
