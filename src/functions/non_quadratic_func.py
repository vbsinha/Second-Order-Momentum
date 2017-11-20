from function import Function
import numpy as np
        
class NonQuadraticFunc(Function):
    
    def f(self, x):
        assert x.shape[0] == 2, "Passed point is not 2 dimensional"
        first_power = x[0] + 3*x[1] - 0.1
        second_power = x[0] - 3*x[1] - 0.1
        third_power = -x[0] - 0.1
        return np.exp(first_power) + np.exp(second_power) + np.exp(third_power)
        
    def grad_f(self, x):
        assert x.shape[0] == 2, "Passed point is not 2 dimensional"
        first_power = x[0] + 3*x[1] - 0.1
        second_power = x[0] - 3*x[1] - 0.1
        third_power = -x[0] - 0.1
        grad_f_x1 = np.exp(first_power) + np.exp(second_power) - np.exp(third_power)
        grad_f_x2 = 3*np.exp(first_power) - 3*np.exp(second_power)
        return np.array([grad_f_x1, grad_f_x2])
        
    def grad2_f(self, x):
        assert x.shape[0] == 2, "Passed point is not 2 dimensional"
        first_power = x[0] + 3*x[1] - 0.1
        second_power = x[0] - 3*x[1] - 0.1
        third_power = -x[0] - 0.1
        grad_f_x11 = np.exp(first_power) + np.exp(second_power) + np.exp(third_power)
        grad_f_x21 = 3*np.exp(first_power) - 3*np.exp(second_power)
        grad_f_x12 = 3*np.exp(first_power) -3* np.exp(second_power)
        grad_f_x22 = 9*np.exp(first_power) + 9*np.exp(second_power)
        return np.array([[grad_f_x11, grad_f_x12], [grad_f_x21, grad_f_x22]])
        
    def __str__(self):
        return "Non Quadratic Function "
