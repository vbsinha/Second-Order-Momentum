import numpy as np
from function import Function

class CubicRegFunc(Function):
    
    def f(self, x):
        assert x.shape[0] == 2, "Passed point is not 2 dimensional"
        return x[0]**2 * x[1]**2 + x[0]**2 + x[1]**2
        
    def grad_f(self, x):
        assert x.shape[0] == 2, "Passed point is not 2 dimensional"
        grad_f_x1 = 2*x[0]*x[1]**2 + 2*x[0]
        grad_f_x2 = 2*x[1]*x[0]**2 + 2*x[1]
        return np.array([grad_f_x1, grad_f_x2])
        
    def grad2_f(self, x):
        assert x.shape[0] == 2, "Passed point is not 2 dimensional"
        second_term = 4*x[0]*x[1]
        grad2_f_x1_x1 = 2*x[1]**2 + 2
        # grad2_f_x1_x2 = second_term
        # grad2_f_x2_x1 = second_term
        grad2_f_x2_x2 = 2*x[0]**2 + 2
        return np.array([[grad2_f_x1_x1, second_term], [second_term, grad2_f_x2_x2]])
        
    def fstar(self):
        return 0
        
    def l(self):
        return 2
        
    def __str__(self):
        return "Ryan Function"
