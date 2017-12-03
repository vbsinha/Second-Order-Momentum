import numpy as np
from function import Function 

class CustomFunction(Function):
    
    def __init__(self,f,grad_f,grad2_f=None,f_star=None,f_L=None,name=None):
        self.f_obj = f
        self.grad_f_obj = grad_f 
        if grad2_f != None:
            self.grad2_f_obj = grad2_f
        if f_star != None:
            self.fstar_obj = f_star
        if f_L != None:
            self.L_obj = f_L
        self.name = name
        
    def f(self,x):
        return self.f_obj(x)
        
    def grad_f(self,x):
        return self.grad_f_obj(x)
        
    def grad2_f(self,x):
        return self.grad2_f_obj(x)
        
    def fstar(self):
        return self.fstar_obj 
        
    def L(self):
        return self.L_obj
    
    def __str__(self):
        if self.name != None:
            return name 
        else:
            return "No name given"
