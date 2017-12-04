import numpy as np

class Function:
    
    def f(self, x):
        raise NotImplementedError("A derived class must be used")
     
    def grad_f(self, x):
        raise NotImplementedError("A derived class must be used")
        
    def grad2_f(self, x):
        raise NotImplementedError("A derived class must be used")
        
    def assert_dimensions(self,x,size):
    	assert x.shape[0] == 2, "Passed point is not 2-dimensional"
        
    def __str__(self):
        print "WARNING : This function does not have a name"
        return "Function"
        
    def fstar(self):
        print "WARNING: This function does not have a fstar defined"
        return 0
        
    def L(self):
        print "WARNING: This function does not have L defined"
        return 0
   
    def levels(self):
        print "WARNING: This function does not have levels defined"
        return np.np.linspace(0, 20, 15)

    def domain(self):
        print "WARNING: This function does not have domain defined"
        return [[-10,10],[-10,10]]
