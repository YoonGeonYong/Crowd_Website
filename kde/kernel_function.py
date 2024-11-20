import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


''' kernel function (K)'''
def gaussian(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-np.square(x) / 2)

def epanechnikov(x):
    return np.where(np.abs(x) < np.sqrt(5),
                    3/(4*np.sqrt(5)) * (1 - (1/5) * np.square(x)), 0)

def biweight(x):
    return np.where(np.abs(x) < 1,
                    (15/16) * np.square(1 - np.square(x)), 0)

def triangular(x):
    return np.where(np.abs(x) < 1,
                    1 - np.abs(x), 0)

def rectangular(x):
    return np.where(np.abs(x) < 1,
                    1/2, 0)



''' plot '''
def plot(K):
    x = np.arange(-3, 3, 0.01)  # 그래프
    y = K(x)

    area, _ = quad(K, -10, 10)  # 적분
    print(area)

    plt.axes().set_aspect('equal')
    plt.xlim([-2,2])
    plt.ylim([0,2])

    plt.plot(x, y, label=f'{K.__name__}')
    plt.fill_between(x, y, alpha=0.5)

    plt.legend()
    plt.show()