import numpy as np

epsilon = 1e-15

class GradientDescent:
	
    def __init__(self,function,method,momentum_function='momentum',gamma=0):
        self.function = function 
        self.method = method 
        assert momentum_function in ['momentum','nesterov'] 
        self.momentum_function = momentum_function
        self.gamma = gamma
        self.velocity_update_dict = {'momentum': self.momentum_velocity_update,
									'nesterov': self.nesterov_velocity_update}
        self.position_update_dict = {'momentum': self.momentum_position_update,
									'nesterov': self.nesterov_position_update}
		
    def get_update_before_momentum(self,x):
        return self.method(self.function,x)
		
    def momentum_velocity_update(self,x,v):
        update_before_momentum = self.get_update_before_momentum(x)
        update = self.gamma*v + (1-self.gamma)*update_before_momentum
        return update 
		
    def nesterov_velocity_update(self,x,v):
        update = self.gamma*v + (1-self.gamma)*x
        return update
		
    def momentum_position_update(self,x,v,eta):
        velocity_update = self.momentum_velocity_update(x,v)
        update = x - eta*velocity_update
        return update 
		
    def nesterov_position_update(self,x,v,eta):
        velocity_update = self.nesterov_velocity_update(x,v)
        update_before_momentum = self.get_update_before_momentum(velocity_update)
        update = velocity_update - eta*update_before_momentum
        return update
		
    def get_velocity_update(self,x,v):
	    return self.velocity_update_dict[self.momentum_function](x,v)
		
    def get_position_update(self,x,v,eta):
        return self.position_update_dict[self.momentum_function](x,v,eta)
		
    def gradient_descent(self,start_x,num_iterations,step_size):
        x = start_x
        v = 0
        points = [start_x]
        for i in range(0,num_iterations):
            if np.fabs(self.function.f(x) - self.function.fstar()) < epsilon:
                break
            old_v = v
            old_x = x
            v = self.get_velocity_update(x,v)
            if self.momentum_function == 'momentum':
                eta = step_size(i,x,v,self.function)
            else:
                v_method_update = self.method(self.function,v)
                eta = step_size(i,v,v_method_update,self.function)
            x = self.get_position_update(x,old_v,eta)
            points.append(x)
            self.method.update_state(self.function, x, old_x) 
        points = np.array(points)
        return points, x
