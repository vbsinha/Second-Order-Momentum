from function import Function
import numpy as np
        
class BiggsEXP2Func(Function):
    
    def f(self, x):
        self.assert_dimensions(x,2)
        result = 0
        for i in range(1,11):
        	t_i = 0.1*i
        	y_i = np.exp(-t_i) - 5*np.exp(10*t_i)
        	f_i = np.exp(-t_i*x[0]) - 5*np.exp(-t_i*x[1]) - y_i
        	result += np.power(f_i,2)
        return result
        
    def grad_f(self, x):
        self.assert_dimensions(x,2)
        result = np.zeros(2)
        for i in range(1,11):
        	t_i = 0.1*i
        	y_i = np.exp(-t_i) - 5*np.exp(10*t_i)
        	f_i = np.exp(-t_i*x[0]) - 5*np.exp(-t_i*x[1]) - y_i
        	derivative_1 = -t_i*np.exp(-t_i*x[0])
        	derivative_2 = 5*t_i*np.exp(-t_i*x[1])
        	result[0] += 2*f_i*derivative_1
        	result[1] += 2*f_i*derivative_2
        return result
        
    def grad2_f(self, x):
        self.assert_dimensions(x,2)
        print "ERROR: Hessian not implemented for Biggs EXP2 function"
        return np.array([[1, 1], [1, 1]])
        
    def __str__(self):
        return "Biggs EXP2 Function"
        
    def domain(self):
        return [[-20, 20], [-20, 20]]
        
    def levels(self):
        return np.linspace(0, 20, 15)
