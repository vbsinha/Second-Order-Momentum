import numpy as np

class FixedStep:
	
	def __init__(self,eta):
		self.eta = eta
		
	def __call__(self,i,x,function,method):
		return self.eta
		
class BacktrackingLineStep:
	
	def __init__(self,alpha,beta,max_backtracks=float('inf')):
		self.alpha = alpha
		self.beta = beta
		self.max_backtracks = max_backtracks
		
	def __call__(self,i,x,function,method):
		U = method(function,x)
		grad_f = function.grad_f(x)
		t = 1
		lhs = function.f(x-t*grad_f)
		rhs = function.f(x) + self.alpha*t*np.dot(grad_f,-U)
		counter = 0
		while counter < self.max_backtracks and lhs > rhs:
			t = self.beta*t
			lhs = function.f(x-t*grad_f)
			rhs = function.f(x) + self.alpha*t*np.dot(grad_f,-U)
			counter += 1
		return t
		
