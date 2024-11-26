import numpy as np


''' kernel function (K)'''
def gaussian(x):
    return (1 / np.sqrt(2*np.pi)) * np.exp(-np.square(x)/2)

def epanechnikov(x): # parabolic
    return np.maximum((3/4) * (1 - np.square(x)), 0)

def biweight(x): # quartic
    return np.where(np.abs(x) < 1, (15/16) * np.square(1 - np.square(x)), 0)

def triweight(x):
    return np.maximum((35/32) * np.power(1 - np.square(x), 3), 0)

def tricube(x):
    return np.maximum((70/81) * np.power(1 - np.power(np.abs(x), 3), 3), 0)

def cosine(x):
    return np.where(np.abs(x) < 1, (np.pi/4) * np.cos((np.pi/2) * x), 0)

def logistic(x):
    return 1 / (np.exp(x) + 2 + np.exp(-x))

def sigmoid(x):
    return (2/np.pi) * (1 / (np.exp(x) + np.exp(-x)))

def silverman(x):
    return (1/2) * np.exp(-np.abs(x)/np.sqrt(2)) * np.sin((np.abs(x)/np.sqrt(2)) + (np.pi/4))

def triangular(x):
    return np.maximum(1 - np.abs(x), 0)

def rectangular(x): # uniform
    return np.where(np.abs(x) < 1, (1/2), 0)



''' plot '''
def plot():
    import matplotlib.pyplot as plt
    from scipy.integrate import quad

    kernels = [gaussian, epanechnikov, biweight, triweight, tricube, cosine, logistic, sigmoid, silverman, triangular, rectangular]
    x = np.arange(-5, 5, 0.01)

    for k in kernels:
        y = k(x)
        plt.plot(x, y, label=f'{k.__name__}')

        area, _ = quad(k, -20, 20)  # 적분
        print(f'{k.__name__} ', area)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot()