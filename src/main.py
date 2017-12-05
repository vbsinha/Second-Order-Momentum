import methods
import numpy as np
from plt import plot, plot_convergence_rate
from functions.quadratic import QuadraticFunc
from functions.ryanfunc import RyanFunc
from functions.bealefunc import BealeFunc
from functions.cubicregfunc import CubicRegFunc
from functions.non_quadratic_func import NonQuadraticFunc
from functions.examplefunc import ExampleFunc
from functions.bird_function import BirdFunc
from functions.levi_function import LevyFunc
from functions.biggs_exp2 import BiggsEXP2Func
from functions.rosenbrockfunc import RosenbrockFunc
from functions.logistic import LogisticFunc
from gd import GradientDescent
from step_sizes import *

epsilon = 1e-15

q = QuadraticFunc()
x = q.start_x()  
g_first = GradientDescent(q, methods.FirstOrder())
g_first_mom = GradientDescent(q, methods.FirstOrder(), 'nesterov', 0.9)
g_second = GradientDescent(q, methods.SecondOrder())
g_second_mom = GradientDescent(q, methods.SecondOrder(), 'nesterov', 0.9)
g_bfgs = GradientDescent(q, methods.BFGS(np.identity(2)))
g_bfgs_mom = GradientDescent(q, methods.BFGS(np.identity(2)), 'nesterov', 0.9)
# g_cr = GradientDescent(q, methods.CubicRegularization())
# print x
points_first, x_first = g_first.gradient_descent(x, 500, FixedStep(0.1))
points_first_mom, x_first_mom = g_first_mom.gradient_descent(x, 500, FixedStep(0.1))
points_second, x_second = g_second.gradient_descent(x, 80, FixedStep(1))
points_second_mom, x_second_mom = g_second_mom.gradient_descent(x, 500, FixedStep(1))
points_bfgs, x_bfgs = g_bfgs.gradient_descent(x, 500, FixedStep(1))
points_bfgs_mom, x_bfgs_mom = g_bfgs_mom.gradient_descent(x, 500, FixedStep(1))
# points_cr, x_cr = g_cr.gradient_descent(x, 100, FixedStep(-1))
print "Minima achieved at (without momentum) : ", x_first
print "Minima achieved at (with momentum) : ", x_first_mom
print "Minima achieved at for second order (without momentum) : ", x_second
print "Minima achieved at for second order (with momentum) : ", x_second_mom
print "Minima achieved at for bfgs (without momentum) : ", x_bfgs
print "Minima achieved at for bfgs (with momentum) : ", x_bfgs_mom
# print "Minima achieved at for Cubic Regularization (without momentum) : ", x_cr
plot(q, q.domain()[0], q.domain()[1], [(points_first, "First order"), (points_first_mom, "First order with momentum"), (points_bfgs, "BFGS without momentum"), (points_bfgs_mom, "BFGS with momentum"),(points_second, "Second order"), (points_second_mom, "Second order with momentum")]) #, (points_cr, "Cubic Regularization") ])
plot_convergence_rate(q, [0,500], [(points_first, "First order"), (points_first_mom, "First order with momentum"), (points_bfgs, "BFGS without momentum"), (points_bfgs_mom, "BFGS with momentum"),(points_second, "Second order"), (points_second_mom, "Second order with momentum")]) #, (points_cr, "Cubic Regularization")])
