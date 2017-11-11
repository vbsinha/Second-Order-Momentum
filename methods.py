import numpy as np

def first_order(self,function,x):
	gradient = function.grad_f(x)
	return gradient 
		
def second_order(self,function,x):
	grad_2 = function.grad2_f(x)
	grad_1 = function.grad_f(x)
	grad_inv = np.linalg.inv(grad_1)
	update = np.matmul(grad_2,grad_inv)
	return update 
	

		
