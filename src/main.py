import methods
import numpy as np
from plt import plot
from functions.quadratic import QuadraticFunc
from functions.ryanfunc import RyanFunc
from functions.non_quadratic_func import NonQuadraticFunc
from gd import GradientDescent

x = np.array([6, 10])
q = NonQuadraticFunc()
g_first = GradientDescent(q, methods.first_order, 'momentum', 0)
g_first_mom = GradientDescent(q, methods.first_order, 'momentum', 0.9)
g_second = GradientDescent(q, methods.second_order, 'momentum', 0)
g_second_mom = GradientDescent(q, methods.second_order, 'momentum', 0.9)
points_first, x_first = g_first.gradient_descent(x, 1000, 0.01)
points_first_mom, x_first_mom = g_first_mom.gradient_descent(x, 1000, 0.01)
points_second, x_second = g_second.gradient_descent(x, 1000, 0.01)
points_second_mom, x_second_mom = g_second_mom.gradient_descent(x, 1000, 0.01)
print "Minima achieved at (without momentum) : ", x_first
print "Minima achieved at (with momentum) : ", x_first_mom
print "Minima achieved at for second order (without momentum) : ", x_second
print "Minima achieved at for second order (with momentum) : ", x_second_mom
plot(q, [-10, 10], [-20, 20], [(points_first, "First order"), (points_first_mom, "First order with momentum"), (points_second, "Second order"), (points_second_mom, "Second order with momentum")])
