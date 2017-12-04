from function import Function
import numpy as np
        
class QuadraticFunc(Function):
    
    def f(self, x):
        self.assert_dimensions(x,2)
        return 1.125*x[0]*x[0] + 0.5*x[0]*x[1] + 0.75*x[1]*x[1] + 2*x[0] + 2*x[1]
        
    def grad_f(self, x):
        self.assert_dimensions(x,2)
        grad_f_x1 = 2.25*x[0] + 0.5*x[1] + 2
        grad_f_x2 = 0.5*x[0] + 1.5*x[1] + 2
        return np.array([grad_f_x1, grad_f_x2])
        
    def grad2_f(self, x):
        self.assert_dimensions(x,2)
        return np.array([[2.5, 0.5], [0.5, 1.5]])
        
    def __str__(self):
        return "Quadratic Function"
        
    def fstar(self):
        return -1.76
        
    def levels(self):
        return np.linspace(0, 20, 15)
        
    def domain(self):
        return [[-4.5, 4.5],[-4.5, 4.5]]
        
    def start_x(self):
        return np.array([4, 1])
