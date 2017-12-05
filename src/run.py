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
from gd import GradientDescent
from step_sizes import *

# function_list = [BealeFunc(), BirdFunc(), LevyFunc(), RosenbrockFunc(), BiggsEXP2Func(), QuadraticFunc(), RyanFunc(), CubicRegFunc(), NonQuadraticFunc()]
function_list = [RosenbrockFunc()]

k_list = [0,5,10,20,50,100,250,500]

epsilon_list = [1,1e-1,1e-2,1e-3,1e-5,1e-8,1e-12,1e-15]

for objective in function_list:
    q = objective
    x = q.start_x()  
    g_first = GradientDescent(q, methods.FirstOrder())
    g_first_mom = GradientDescent(q, methods.FirstOrder(), 'momentum', 0.9)
    g_first_nest = GradientDescent(q, methods.FirstOrder(), 'nesterov', 0.9)
    g_second = GradientDescent(q, methods.SecondOrder())
    g_second_mom = GradientDescent(q, methods.SecondOrder(), 'momentum', 0.9)
    g_second_nest = GradientDescent(q, methods.SecondOrder(), 'nesterov', 0.9)
    g_bfgs = GradientDescent(q, methods.BFGS(np.identity(2)))
    g_bfgs_mom = GradientDescent(q, methods.BFGS(np.identity(2)), 'momentum', 0.9)
    g_bfgs_nest = GradientDescent(q, methods.BFGS(np.identity(2)), 'nesterov', 0.9)
    # g_cr = GradientDescent(q, methods.CubicRegularization())
    # print x
    print "Function : ", str(objective)
    print "Start point : ", "[2, 3]"
    print "\\hline"
    print "K & Gradient Descent & Gradient Descent with Momentum & Gradient Descent with Nesterov Momentum & Newton's Method & Newton's Method with Momentum & Newton's Momentum with Nesterov Momentum & BFGS & BFGS with Momentum & BFGS with Nesterov Momentum \\\\"
    print "\\hline"
    for k in k_list:
        points_first, x_first = g_first.gradient_descent(x, k, FixedStep(0.001))
        points_first_mom, x_first_mom = g_first_mom.gradient_descent(x, k, FixedStep(0.001))
        points_first_nest, x_first_nest = g_first_nest.gradient_descent(x, k, FixedStep(0.001))
        points_second, x_second = g_second.gradient_descent(x, k, FixedStep(1))
        points_second_mom, x_second_mom = g_second_mom.gradient_descent(x, k, FixedStep(1))
        points_second_nest, x_second_nest = g_second_nest.gradient_descent(x, k, FixedStep( 1))
        points_bfgs, x_bfgs = g_bfgs.gradient_descent(x, k, FixedStep(0.0001))
        points_bfgs_mom, x_bfgs_mom = g_bfgs_mom.gradient_descent(x, k, FixedStep(0.0001))
        points_bfgs_nest, x_bfgs_nest = g_bfgs_mom.gradient_descent(x, k, FixedStep(0.0001))
        # points_cr, x_cr = g_cr.gradient_descent(x, 100, FixedStep(-1))1
    #    print "Minima achieved at (without momentum) : ", x_first
    #    print "Minima achieved at (with momentum) : ", x_first_mom
    #    print "Minima achieved at for second order (without momentum) : ", x_second
    #    print "Minima achieved at for second order (with momentum) : ", x_second_mom
    #    print "Minima achieved at for bfgs (without momentum) : ", x_bfgs
    #    print "Minima achieved at for bfgs (with momentum) : ", x_bfgs_mom
        # print "Minima achieved at for Cubic Regularization (without momentum) : ", x_cr
        print k,"&",objective.f(x_first) - objective.fstar(),"&",objective.f(x_first_mom) - objective.fstar(),"&",objective.f(x_first_nest) - objective.fstar(),"&",objective.f(x_second) - objective.fstar(),"&",objective.f(x_second_mom) - objective.fstar(),"&",objective.f(x_second_nest) - objective.fstar(),"&",objective.f(x_bfgs) - objective.fstar(),"&",objective.f(x_bfgs_mom) - objective.fstar(),"&",objective.f(x_bfgs_nest) - objective.fstar(),"\\\\"
        print "\\hline"
 #   plot(q, q.domain()[0], q.domain()[1], [(points_first, "First order"), (points_first_mom, "First order with momentum"), (points_bfgs, "BFGS without momentum"), (points_bfgs_mom, "BFGS with momentum"),(points_second, "Second order"), (points_second_mom, "Second order with momentum")])#, (points_cr, "Cubic Regularization") ])
 #   plot_convergence_rate(q, [0,100], [(points_first, "First order"), (points_first_mom, "First order with momentum"), (points_bfgs, "BFGS without momentum"), (points_bfgs_mom, "BFGS with momentum"),(points_second, "Second order"), (points_second_mom, "Second order with momentum")])#, (points_cr, "Cubic Regularization")])




