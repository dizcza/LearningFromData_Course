import random
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import inv
import math


sign = lambda x: math.copysign(1, x)

def get_rand_coef():
    """ Returns random coef a and b from X = [-1, 1]x[-1, 1]. """
    X_target = [[], []]
    for i in range(2):
        X_target[0].append(random.randrange(-100, 100, 1) / 100.0)
        X_target[1].append(random.randrange(-100, 100, 1) / 100.0)

    # y = ax + b
    dx = X_target[0][0] - X_target[0][1]
    if dx == 0.0:       # if denominator == 0
        dx = 10 ** -3
    a = (X_target[1][0] - X_target[1][1]) / dx
    b = X_target[1][0] - a * X_target[0][0]
    return a, b


def display(X, Y, w, a, b):
    """ Display all. """
    N = X.shape[0]
    x0 = X[:, 1]    # as simple x in plot

    # plotting target line (hidden function)
    x_target = np.array(range(-100, 101)) / 100.0
    y_target = a * x_target + b
    plt.axis([-1, 1, -1, 1])
    plt.plot(x_target, y_target, '--', lw=3)

    # plotting obtained linear classifier
    if abs(w[2]) < 10 ** -3:
        w[2] = 10 ** -3
    y_obt = -w[0]/w[2] - w[1]/w[2] * x0
    plt.plot(x0, y_obt, lw=1, color='black')

    # plotting labeled points
    for i in range(N):
        if Y[i] < 0:
            plt.plot(X[i][1], X[i][2], 'ro')
        else:
            plt.plot(X[i][1], X[i][2], 'go')
    plt.show()


def generate_data(N, a, b):
    """
        Generates a binary data:
            Y = {-1, 1}
            X = [-1, 1] x [-1, 1].
    """
    X = []
    Y = []
    for i in range(N):
        x0 = random.randrange(-100, 100, 1) / 100.0
        x1 = random.randrange(-100, 100, 1) / 100.0
        X.append([1.0, x0, x1])
        delta = x1 - (a * x0 + b)
        Y.append(sign(delta))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def compute_weights(X, Y):
    N = X.shape[0]
    Xtr = X.T
    Xpseudo = inv(Xtr.dot(X)).dot(Xtr)
    W = Xpseudo.dot(Y)
    return W


def compute_error(X, Y, W):
    N = X.shape[0]
    num_of_misclassified = 0.
    for i in range(N):
        binary_val = sign(W.T.dot(X[i].T))
        if binary_val * Y[i] < 0:
            num_of_misclassified += 1
    return float(num_of_misclassified / N)


aver_error = 0
N_in = 100
N_out = 1000
iterations_need = 1000
for dummy_ind in range(iterations_need):
    a, b = get_rand_coef()
    X_in, Y_in = generate_data(N_in, a, b)
    #X_out, Y_out = generate_data(N_out, a, b)
    W = compute_weights(X_in, Y_in)
    error_in = compute_error(X_in, Y_in, W)
    #error_out = compute_error(X_out, Y_out, W)
    #display(X_out, Y_out, W, a, b)
    #display(X_in, Y_in, W, a, b)
    aver_error += error_in / iterations_need

print aver_error

#------------------------
# RESULTS:
#   #5 (c): error_in  == 0.040 - closest to 0.01
#   #6 (C): error_out == 0.048 - closest to 0.01 (I can argue)
#   #7: (a)
#   #8: (d) - LRC error will be a half of samples (a bit smaller)
#   #9: (a) - rewrite inequality by making a1 * x1^2 + a2 * x2^2 < R^2 == 1
#   #10: (b) - Eout ~ Ein (as a result from #5 and #6); Ein == 0.132 (a result from #2: nu=0.04 (from #5), lambda=0.9)
