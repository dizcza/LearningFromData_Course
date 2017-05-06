from math import factorial, exp, log, sqrt, sin, pi
import matplotlib.pyplot as plt
import numpy as np


def display(X, Y, my_coef):
    plt.plot(X, Y, 'ro')
    X_dots = np.linspace(-1, 1, 51)
    Y_sin_dots = [sin(pi * x) for x in X_dots]
    Y_linear_dots = [my_coef * x for x in X_dots]
    plt.plot(X_dots, Y_sin_dots, 'b', lw=1)
    plt.plot(X_dots, Y_linear_dots, 'g', lw=2)
    plt.show()


def linear_reg_kernel():
    X = []
    Y = []
    for i in range(2):
        rand_x = np.random.randint(-1000, 1001) / 1000.0
        X.append(rand_x)
        Y.append(sin(pi * rand_x))
    X = np.array(X)
    Y = np.array(Y)
    coef = X.dot(Y) / X.dot(X)
    #coef = 1.426
    #X = X[:,np.newaxis]
    #a, _, _, _ = np.linalg.lstsq(X, Y)

    #display(X, Y, coef)
    return coef


def estimate_linear_coef():
    alpha_list = []
    num_of_iter = 10000
    for i in range(num_of_iter):
        current_coef = linear_reg_kernel()
        alpha_list.append(current_coef)
    coef = sum(alpha_list) / num_of_iter
    print "Average LR coef: %g" % coef
    return coef, alpha_list


def evaluate_bias():
    coef = 1.426    # got from estimate_linear_coef()
    X_dots = np.linspace(-1, 1, 51)
    N = len(X_dots)
    Y_sin_dots = [sin(pi * x) for x in X_dots]
    Y_linear_dots = [coef * x for x in X_dots]
    integral = 0.0
    dx_step = 2.0 / (len(X_dots) - 1)
    for i in range(N):
        dy = (Y_sin_dots[i] - Y_linear_dots[i]) ** 2
        integral += dy * dx_step
    bias = integral / 2.0
    print "Bias val: %g" % bias
    # returns 0.31
    return bias


def evaluate_variance():
    coef_average, coef_list = estimate_linear_coef()
    K = float(len(coef_list))
    X_dots = np.linspace(-1, 1, 51)
    dx_step = 2.0 / (len(X_dots) - 1)
    variance = 0.0
    for x in X_dots:
        y_dot = [abs((coef_average - alpha) * x) for alpha in coef_list]
        y_deviation = sum(y_dot) / K
        var_x = x ** 2 / K * sum([(alpha - coef_average) ** 2 for alpha in coef_list])
        variance += var_x * dx_step
    normalized_variance = variance / 2.0
    print "Variance: %g" % normalized_variance
    # returns 0.25
    return normalized_variance

evaluate_variance()
