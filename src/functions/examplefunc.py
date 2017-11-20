from function import Function
import numpy as np
        
class QuadraticFunc(Function):
    
    def f(self,x):
    	assert x.shape[0] == 2, "Passed point is not 2 dimensional"
		x1 = x[0]
		x2 = x[1]
		obj = x1**2 - 2.0 * x1 * x2 + 4 * x2**2
		return obj

	# define objective gradient
	def grad_f(self,x):
		assert x.shape[0] == 2, "Passed point is not 2 dimensional"
		x1 = x[0]
		x2 = x[1]
		grad = []
		grad.append(2.0 * x1 - 2.0 * x2)
		grad.append(-2.0 * x1 + 8.0 * x2)
		return grad
        
    def grad2_f(self, x):
        assert x.shape[0] == 2, "Passed point is not 2 dimensional"
        print "ERROR: Hessian not implemented for example function"
        return np.array([[0, 0], [0, 0]])
        
    def __str__(self):
        return "Example Function from http://apmonitor.com/me575/index.php/Main/QuasiNewton "
