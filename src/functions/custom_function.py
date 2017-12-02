import numpy as np
from function import Function 

class CustomFunction(Function):
    
    def __init__(self,f,grad_f,grad2_f=None,f_star=None,f_L=None,name=None):
        self.f = f
        self.grad_f = grad_f 
        if grad2_f != None:
            self.grad2_f = grad2_f
        if f_star != None:
            self.fstar = f_star
        if f_L != None:
            self.L = f_L
        self.name = name
    
    def __str__(self):
        if self.name != None:
            return name 
        else:
            return "No name given"
