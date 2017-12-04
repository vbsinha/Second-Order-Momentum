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
from gd import GradientDescent
from step_sizes import *

function_list = [BealeFunc(), BirdFunc(), LevyFunc(), RosenbrockFunc(), BiggsEXP2Func(), QuadraticFunc(), RyanFunc(), CubicRegFunc(), NonQuadraticFunc()]

k_list = [0,5,10,20,50,100,250,500]

epsilon_list = [1,1e-1,1e-2,1e-3,1e-5,1e-8,1e-12,1e-15]






