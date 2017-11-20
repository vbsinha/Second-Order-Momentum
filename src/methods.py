import numpy as np

class FirstOrder:
	
	def __call__(self,function,x):
		gradient = function.grad_f(x)
		return gradient 
		
class SecondOrder:
	
	def __call__(self,function,x):
		grad_2 = function.grad2_f(x)
		grad_1 = function.grad_f(x)
		grad_inv = np.linalg.inv(grad_2)
		update = np.matmul(grad_inv,grad_1)
		return update
	
class BFGS:
	
	def __init__(self,initial_H):
		self.H = initial_H
		
	def __call__(self,function,x):
		pass
