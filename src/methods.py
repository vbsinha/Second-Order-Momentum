import numpy as np

def first_order(function,x):
	gradient = function.grad_f(x)
	return gradient 
		
def second_order(function,x):
	grad_2 = function.grad2_f(x)
	grad_1 = function.grad_f(x)
	grad_inv = np.linalg.inv(grad_2)
	update = np.matmul(grad_inv,grad_1)
	return update 
	

		
