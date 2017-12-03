# Baele's function
# Evaluate in range : [-4.5, 4.5]

import numpy as np
from function import Function


class BaeleFunc(Function):
    
    def f(self, x):
        assert x.shape[0] == 2, "Passed point is not 2 dimensional"
        return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2
        
    def grad_f(self, x):
        assert x.shape[0] == 2, "Passed point is not 2 dimensional"
        t1	= 2*(1.5 - x[0] + x[0]*x[1])
	    t2	= 2*(2.25 - x[0] + x[0]*x[1]**2)
	    t3	= 2*(2.625 - x[0] + x[0]*x[1]**3)
        grad_f_x1 = t1*(-1 + x[1]) + t2*(-1 + x[1]**2) + t3*(-1 + x[1]**3)
        grad_f_x2 = t1*x[0] + t2*2*x[0]*x[1] + t3*3*x[0]*x[1]**2
        return np.array([grad_f_x1, grad_f_x2])
        
    def grad2_f(self, x):
        assert x.shape[0] == 2, "Passed point is not 2 dimensional"
        t1	= 2*(1.5 - x[0] + x[0]*x[1])
	    t2	= 2*(2.25 - x[0] + x[0]*x[1]**2)
	    t3	= 2*(2.625 - x[0] + x[0]*x[1]**3)
	    a1 = (-1 + x[1])
	    a2 = (-1 + x[1]**2)
	    a3 = (-1 + x[1]**3)
        second_term = t1 + a1*x[0] + t2*2*x[1] + 2*a2*2*x[0]*x[1] + t3*3*x[1]**2 + 2*a3*3*x[0]*x[1]**2
        grad2_f_x1_x1 = 2(a1**2 + a2**2 + a3**2)
        # grad2_f_x1_x2 = -1*second_term
        # grad2_f_x2_x1 = -1*second_term
        grad2_f_x2_x2 = 2*x[0]**2 + t2*2*x + 2*(2*x[0]*x[1])**2 + t3*3*2*x[0]*x[1] + 6*(x[1]**2*x[0])
        return np.array([[grad2_f_x1_x1, second_term], [second_term, grad2_f_x2_x2]])
        
    def fstar(self):
        return 0
        
    def __str__(self):
        return "Baele Function"
        
