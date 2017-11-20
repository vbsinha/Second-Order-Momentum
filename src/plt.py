import matplotlib.pyplot as plt
import numpy as np

def plot(func, x_range, y_range, method_points):
    r = np.linspace(x_range[0], x_range[1], 100)
    s = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(r, s)
    Z = []
    for i in xrange(len(X)):
        Z.append(func.f(np.array([X[i], Y[i]])))
    cp = plt.contour(X, Y, Z, levels=np.linspace(0, 500, 20), cmap=plt.cm.jet)
    for points, l in method_points:
        plt.plot(points[:,0], points[:,1], label=l)
    plt.title(str(func))  
    plt.legend()  
    plt.show()
