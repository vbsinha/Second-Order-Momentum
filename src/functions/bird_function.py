import numpy as np
from numpy import sin,cos,exp,power
from function import Function

class BirdFunc(Function):
    
    def f(self, x):
        self.assert_dimensions(x,2)
        return np.sin(x[0])*np.exp(np.power(1-np.cos(x[1]),2)) + np.cos(x[1])*np.exp(np.power(1-np.sin(x[0]),2)) + np.power(x[0]-x[1],2)
        
    def grad_f(self, x_vec):
        self.assert_dimensions(x_vec,2)
        x = x_vec[0]
        y = x_vec[1]
        grad_f_x1 = 2*x - 2*y + exp(power(cos(y) - 1,2))*cos(x) + 2*exp(power(sin(x) - 1,2))*cos(x)*cos(y)*(sin(x) - 1)
        grad_f_x2 = 2*y - 2*x - exp(power(sin(x) - 1,2))*sin(y) - 2*exp(power(cos(y) - 1,2))*sin(x)*sin(y)*(cos(y) - 1)
        return np.array([grad_f_x1, grad_f_x2])
        
    def grad2_f(self, x_vec):
        self.assert_dimensions(x_vec,2)
        x = x_vec[0]
        y = x_vec[1]
        grad2_f_x1_x1 = 2*exp(power(sin(x) - 1,2))*power(cos(x),2)*cos(y) - exp(power(cos(y) - 1,2))*sin(x) + 4*exp(power(sin(x) - 1,2))*power(cos(x),2)*cos(y)*power(sin(x) - 1,2) - 2*exp(power(sin(x) - 1,2))*cos(y)*sin(x)*(sin(x) - 1) + 2
        grad2_f_x1_x2 = -2*exp(power(cos(y) - 1,2))*cos(x)*sin(y)*(cos(y) - 1) - 2*exp(power(sin(x) - 1,2))*cos(x)*sin(y)*(sin(x) - 1) - 2

        grad2_f_x2_x1 = -2*exp(power(cos(y) - 1,2))*cos(x)*sin(y)*(cos(y) - 1) - 2*exp(power(sin(x) - 1,2))*cos(x)*sin(y)*(sin(x) - 1) - 2

        grad2_f_x2_x2 = 2*exp(power(cos(y) - 1,2))*sin(x)*power(sin(y),2) - exp(power(sin(x) - 1,2))*cos(y) + 4*exp(power(cos(y) - 1,2))*sin(x)*power(sin(y),2)*power(cos(y) - 1,2) - 2*exp(power(cos(y) - 1,2))*cos(y)*sin(x)*(cos(y) - 1) + 2
        return np.array([[grad2_f_x1_x1, grad2_f_x1_x2], [grad2_f_x2_x1, grad2_f_x2_x2]])
        
    def fstar(self):
        return -106.764537
        
    def __str__(self):
        return "Bird Function"
    
    def domain(self):
        return [[-6.28, 6.28],[-6.28, 6.28]]
        
    def levels(self):
        return np.linspace(-90, 40, 15)
        
    def start_x(self):
        return np.array([4, 1])
