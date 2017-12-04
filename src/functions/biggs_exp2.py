from function import Function
import numpy as np
from numpy import exp,power
        
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
        
    def grad2_f(self, x_vec):
        self.assert_dimensions(x_vec,2)
        x = x_vec[0]
        y = x_vec[1]
        result = np.zeros((2,2))
        for i in range(1,11):
            t_i = 0.1*i
            y_i = exp(-t_i) - 5*exp(10*t_i)
            result[0][0] += 2*power(t_i,2)*exp(-2*t_i*x) - 2*power(t_i,2)*exp(-t_i*x)*(y_i - exp(-t_i*x) + 5*exp(-t_i*y))
            result[0][1] += -10*power(t_i,2)*exp(-t_i*x)*exp(-t_i*y)
            result[1][0] += -10*power(t_i,2)*exp(-t_i*x)*exp(-t_i*y)
            result[1][1] += 50*power(t_i,2)*exp(-2*t_i*y) + 10*power(t_i,2)*exp(-t_i*y)*(y_i - exp(-t_i*x) + 5*exp(-t_i*y))
        return result
        
    def fstar(self):
        return 0
        
    def domain(self):
        return [[0,20],[0,20]]
        
    def __str__(self):
        return "Biggs EXP2 Function"
        
    def levels(self):
        return np.linspace(0, 20, 15)
