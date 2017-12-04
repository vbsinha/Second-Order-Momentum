import matplotlib.pyplot as plt
import numpy as np

epsilon = 1e-15

def plot(func, x_range, y_range, method_points):
    r = np.linspace(x_range[0], x_range[1], 100)
    s = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(r, s)
    Z = []
    for i in xrange(len(X)):
        Z.append(func.f(np.array([X[i], Y[i]])))
    cp = plt.contour(X, Y, Z, levels=func.levels(), cmap=plt.cm.jet)
    for points, l in method_points:
        plt.plot(points[:,0], points[:,1], label=l)
    plt.title(str(func))  
    plt.legend()  
    plt.show()
    
def plot_convergence_rate(func, k_range, method_points):
    k_min = k_range[0]
    k_max = k_range[1]
    assert k_min >= 0, "Negative value for iteration given"
    for points, l in method_points:
        plot_points = []
        for i in xrange(k_min,k_max):
            if i >= len(points):
                break
            f_x = func.f(points[i])
            f_star = func.fstar()
   #         if f_x-f_star >= 1e-60:
            plot_points.append([f_x-f_star,i])
    #        else:
     #           plot_points.append([1e-60,i])
        plot_points = np.array(plot_points)
        plt.plot(plot_points[:,1],plot_points[:,0], label=l)
    plt.title(str(func))
    plt.legend()
    plt.yscale('log')
    axes = plt.gca()
    axes.set_ylim([epsilon,1e+8])
    plt.show()
