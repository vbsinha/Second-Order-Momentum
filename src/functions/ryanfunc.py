# Implement f(x) = (10*x_1^2 + x_2^2)/2 + 5*log(1+e^{-x_1-x_2})
# f'(x) = 

import numpy as np
from function import Function


class RyanFunc(Function):
    
    def f(self, x):
        assert x.shape[0] == 2, "Passed point is not 2 dimensional"
        return (10*(x[0]**2) + x[1]**2)/2 + 5*np.log(1+np.exp(-1*(x[0]+x[1])))
        
    def grad_f(self, x):
        assert x.shape[0] == 2, "Passed point is not 2 dimensional"
        second_term = 5/(1+np.exp(-1*(x[0]+x[1])))
        grad_f_x1 = 10*x[0] - second_term*np.exp(-1*(x[0]+x[1]))
        grad_f_x2 = x[1] - second_term*np.exp(-1*(x[0]+x[1]))
        return np.array([grad_f_x1, grad_f_x2])
        
    def grad2_f(self, x):
        assert x.shape[0] == 2, "Passed point is not 2 dimensional"
        b = np.exp(-1*(x[0]+x[1]))
        a = 1 + b
        second_term = (-5*a*b + 5*b*b)/a*a
        grad2_f_x1_x1 = 10 - second_term
        # grad2_f_x1_x2 = -1*second_term
        # grad2_f_x2_x1 = -1*second_term
        grad2_f_x2_x2 = 1 - second_term
        return np.array([[grad2_f_x1_x1, -1*second_term], [-1*second_term, grad2_f_x2_x2]])
        
    def __str__(self):
        return "Ryan Function"
