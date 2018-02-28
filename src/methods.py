import numpy as np
from functions.custom_function import CustomFunction
from step_sizes import BacktrackingLineStep, FixedStep
import sys

class FirstOrder:
    
    def __init__(self):
        pass
	
    def __call__(self,function,x):
        gradient = function.grad_f(x)
        return gradient 
	
    def update_state(self, function, x, old_x):
        pass
		
class SecondOrder:
    
    def __init__(self):
        pass
	
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
	    if np.dot(y, s) < 1e-17:
	        # print "WARNING : BFGS Warning y.T * s is below 1e-17, updates to Hessian inverse stopped... "
	        return
	    lhs = I - np.outer(s, y)/np.dot(y, s) 
	    rhs = I - np.outer(y, s)/np.dot(y, s)
	    a = np.matmul(lhs, self.H)
	    a = np.matmul(a, rhs)
	    self.H = a + np.outer(s, s)/np.dot(y, s)
	    return
	    
class CubicRegularization:
    
    def __init__(self):
        self.method = FirstOrder()
        pass 
        
    def update_state(self, function, x, old_x):
        pass 
        
    def objective_f(self,function,x):
        grad_f = function.grad_f(x)
        grad2_f = function.grad2_f(x)
        L = function.L()
        def update_f(y):
            result = np.dot(grad_f,y-x)
            result = result + 0.5*np.dot(np.dot(y-x,grad2_f),y-x)
            result = result + L*np.power(np.linalg.norm(y-x),3)/6
            return result 
        return update_f
        
    def objective_grad_f(self,function,x):
        grad_f = function.grad_f(x)
        grad2_f = function.grad2_f(x)
        L = function.L()
        def update_grad_f(y):
            result = grad_f
            result = result - np.matmul(grad2_f,x)
            result = result + np.matmul(grad2_f,y)
            norm = np.linalg.norm(y-x) 
            result = result + L*norm*(y-x)
            return result 
        return update_grad_f 
        
    def gradient_descent(self,start_x,num_iterations,step_size):
        x = start_x
        points = [start_x]
        for i in range(0,num_iterations):
            old_x = x
            eta = step_size(i,x,self.function,self.method)
            x = x - eta*self.function.grad_f(x)
            points.append(x)
            self.method.update_state(self.function, x, old_x)
        points = np.array(points)
        return points, x
        
    def __call__(self,function,x):
        old_x = x
        update_f = self.objective_f(function,x)
        update_grad_f = self.objective_grad_f(function,x)
        self.function = CustomFunction(update_f,update_grad_f,name='Objective for Cubic Regularization')
        points, new_x = self.gradient_descent(x,400,FixedStep(0.001))#BacktrackingLineStep(0.5,0.5,10))
        return new_x - old_x
