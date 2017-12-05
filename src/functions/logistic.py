import numpy as np
from function import Function


class LogisticFunc(Function):
    
    def f(self, x):
        assert x.shape[0] == 2, "Passed point is not 2 dimensional"
        return 0.5*(x[0]*x[0] + x[1]*x[1]) + 50*np.log(1 + np.exp(-0.5*x[1])) + 50*np.log(1 + np.exp(0.2*x[0]))
        
    def grad_f(self, x):
        assert x.shape[0] == 2, "Passed point is not 2 dimensional"
        grad_f_x1 = x[0] + 10*np.exp(0.2*x[0])/(1+np.exp(0.2*x[0]))
        grad_f_x2 = x[1] - 25*np.exp(-0.5*x[1])/(1+np.exp(-0.5*x[1]))
        return np.array([grad_f_x1, grad_f_x2])
        
    def grad2_f(self, x):
        assert x.shape[0] == 2, "Passed point is not 2 dimensional"
        second_term = 0
        grad2_f_x1_x1 = 1 + 10*0.2*np.exp(0.2*x[0])/(1 + np.exp(0.2*x[0]))**2
        grad2_f_x2_x2 = 1 - 25*0.5*np.exp(-0.5*x[1])/(1 + np.exp(-0.5*x[1]))**2
        return np.array([[grad2_f_x1_x1, second_term], [second_term, grad2_f_x2_x2]])
        
    def fstar(self):
        return self.f(np.array([-3.37415807, 3.57874376]))
        
    def __str__(self):
        return "Logisitic Function"
        
    def levels(self):
        return np.linspace(0, 10000, 15)
        
    def domain(self):
        return [[-6, 6],[-6, 6]]
        
    def start_x(self):
        return np.array([2, 3])
        
