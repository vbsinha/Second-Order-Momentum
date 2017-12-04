import numpy as np
from numpy import sin,cos,exp,power,pi
from function import Function

class LevyFunc(Function):
    
    def f(self, x_vec):
        self.assert_dimensions(x_vec,2)
        x = x_vec[0]
        y = x_vec[1]
        return (sin(3*pi*x))**2 + ((x-1)**2)*(1+(sin(3*pi*y))**2) + ((y-1)**2)*(1+(sin(2*pi*y))**2)
        
    def grad_f(self, x_vec):
        self.assert_dimensions(x_vec,2)
        x = x_vec[0]
        y = x_vec[1]
        grad_f_x1 = (2*x - 2)*(power(sin(3*pi*y),2) + 1) + 6*pi*cos(3*pi*x)*sin(3*pi*x)
        grad_f_x2 = (2*y - 2)*(power(sin(2*pi*y),2) + 1) + 6*pi*cos(3*pi*y)*sin(3*pi*y)*power(x - 1,2) + 4*pi*cos(2*pi*y)*sin(2*pi*y)*power(y - 1,2)
        return np.array([grad_f_x1, grad_f_x2])
        
    def grad2_f(self, x_vec):
        self.assert_dimensions(x_vec,2)
        x = x_vec[0]
        y = x_vec[1]
        grad2_f_x1_x1 = 18*power(pi,2)*power(cos(3*pi*x),2)-18*power(pi,2)*power(sin(3*pi*x),2)+2*power(sin(3*pi*y),2)+2 

        grad2_f_x1_x2 = 6*pi*cos(3*pi*y)*sin(3*pi*y)*(2*x - 2)

        grad2_f_x2_x1 = 6*pi*cos(3*pi*y)*sin(3*pi*y)*(2*x - 2)

        grad2_f_x2_x2 = 2*power(sin(2*pi*y),2) + 18*power(pi,2)*power(cos(3*pi*y),2)*power(x - 1,2) + 8*power(pi,2)*power(cos(2*pi*y),2)*power(y - 1,2) - 18*power(pi,2)*power(sin(3*pi*y),2)*power(x - 1,2) - 8*power(pi,2)*power(sin(2*pi*y),2)*power(y - 1,2) + 8*pi*cos(2*pi*y)*sin(2*pi*y)*(2*y - 2) + 2

        return np.array([[grad2_f_x1_x1, grad2_f_x1_x2], [grad2_f_x2_x1, grad2_f_x2_x2]])
        
    def fstar(self):
        return 0
        
    def domain(self):
        return [[-10,10],[-10,10]]
        
    def __str__(self):
        return "Levy Function"
