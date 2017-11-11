import numpy as np

class GradientDescent:
	
	def __init__(self,function,method,gamma):
		self.function = function 
		self.method = method 
		self.gamma = gamma
		
	def get_update_before_momentum(self,x):
		return self.method(function,x)
		
	def get_update(self,x,v):
		update_before_momentum = self.get_update_before_momentum(x)
		update = self.gamma*v + (1-self.gamma)*update_before_momentum
		return update 
		
	def gradient_descent(self,start_x,num_iterations,eta):
		x = start_x
		v = 0
		for i in range(0,num_iterations):
			v = self.get_update(x,v)
			x = x - eta*v
		print x
