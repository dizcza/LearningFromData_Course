from math import exp, copysign, log
import matplotlib.pyplot as plt
import numpy as np
import random
from numpy.linalg import norm

def error(u, v):
    return (u * exp(v) - 2 * v * exp(-u)) ** 2


def grad_error(u, v):
    grad_Eu = 2 * (u * exp(v) - 2 * v * exp(-u)) * (exp(v) + 2 * v * exp(-u))
    grad_Ev = 2 * (u * exp(v) - 2 * v * exp(-u)) * (u * exp(v) - 2 * exp(-u))
    return grad_Eu, grad_Ev


def update(u, v, nu_coef):
    u_new = u - nu_coef * grad_error(u, v)[0]
    v_new = v - nu_coef * grad_error(u_new, v)[1]
    #print "before: %f, %f" % (u, v)
    #print "after: %f, %f" % (u_new, v_new)
    return u_new, v_new


def outside_epsilon(x0, x1, eps):
    return abs(2)


def iterate_as_i_think(u, v, epsilon):
    u_before, v_before = u, v
    u, v = update(u_before, v_before, nu_coef)
    iter_num = 1

    print "begin looping..."
    while abs(u - u_before) > epsilon or abs(v - v_before) > epsilon:
        u_before, v_before = u, v
        u, v = update(u_before, v_before, nu_coef)
        iter_num += 1

    diff_u = abs(u - u_before)
    diff_v = abs(v - v_before)
    status = "iter_num: %i; (u_before: %f, u_current: %f) diff is  %e\n\t" % (iter_num, u_before, u, diff_u)
    status += "(v_before: %f, v_current: %f) diff is %e" % (v_before, v, diff_v)
    print status

    return iter_num


def iterate_coord_descent(u, v):
    full_iterations = 15
    for iter_ind in range(full_iterations):
        u, v = update(u, v, nu_coef)
    return error(u, v)


def iterate_error(u, v, epsilon):
    iter_num_unless_error_smaller_than_eps = 0

    while error(u, v) > epsilon:
        u, v = update(u, v, nu_coef)
        iter_num_unless_error_smaller_than_eps += 1
    print "num of iterations unless E < eps: %i" % iter_num_unless_error_smaller_than_eps

    return u, v


#----------------------------------------------------------------------------------------
sign = lambda x: copysign(1, x)

def get_rand_coef():
    """ Returns random coef a and b from X = [-1, 1]x[-1, 1]. """
    X_target = [[], []]
    for i in range(2):
        X_target[0].append(random.randrange(-100, 100, 1) / 100.0)
        X_target[1].append(random.randrange(-100, 100, 1) / 100.0)

    # y = ax + b
    dx = X_target[0][0] - X_target[0][1]
    if dx == 0.0:       # if denominator == 0
        dx = 10 ** -2
    a = (X_target[1][0] - X_target[1][1]) / dx
    b = X_target[1][0] - a * X_target[0][0]
    return a, b


def display(X, Y, w, a, b):
    """ Display all points in X = [-1, 1]x[-1, 1]. """
    N = X.shape[0]
    x0 = X[:, 1]    # as simple x in plot

    # plotting target line (hidden function)
    x_target = np.array(range(-100, 101)) / 100.0
    y_target = a * x_target + b
    plt.axis([-1, 1, -1, 1])
    plt.plot(x_target, y_target, '--', lw=3)

    # plotting obtained linear classifier
    if abs(w[2]) < 10 ** -2:
        w[2] = 10 ** -2
    y_obt = -w[0]/w[2] - w[1]/w[2] * x0
    #print w, x0[N//2], y_obt[N//2], "label: ", Y[N//2]
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


def logistic_error(X, Y, weights):
    # X == (1, x1, x2)
    Nin = X.shape[0]
    E_in = 0
    for i in range(Nin):
        e_in = log(1 + exp(-Y[i] * weights.dot(X[i])))
        E_in += e_in / Nin
    return E_in


def next_epoch(X, Y, weights_prev, nu_coef):
    """ X == (1, x1, x2) x N
        Y = +(-) 1 x N
        Returns [grad_x0, grad_x1, grad_x2].
    """
    Nin, dim = X.shape
    weights = np.copy(weights_prev)

    # concatenate X with Y into XY pairs
    XY_shuffled = np.c_[X, Y]

    # shuffled them
    np.random.shuffle(XY_shuffled)

    X_sh = XY_shuffled[:,:-1]
    Y_sh = XY_shuffled[:, dim]

    grad_E = 0
    for i in range(Nin):
        # compute grad_e for one random point
        grad_e = - Y_sh[i] * X_sh[i] / (1 + exp( Y_sh[i] * weights.dot(X_sh[i]) ))

        # update weights...
        weights -= nu_coef * grad_e

    return weights


def compute_logistic_error(Nin, Nout, nu_coef, epsilon):
    """ Computes weights from the random dataset Xin of size Nin
        and estimates the output logistic error for the new dataset Xout
        of size Nout with respect to weights, obtained from Xin data.
    """
    a, b = get_rand_coef()  # fixed coef. both for Xin and Xout
    Xin, Yin = generate_data(Nin, a, b)

    weights_prev = np.zeros(3)
    weights = next_epoch(Xin, Yin, weights_prev, nu_coef)
    iters_to_converge = 0

    while norm(weights - weights_prev) > epsilon:
        weights_prev = np.copy(weights)
        weights = next_epoch(Xin, Yin, weights_prev, nu_coef)
        iters_to_converge += 1
    #display(Xin, Yin, weights, a, b)

    Xout, Yout = generate_data(Nout, a, b)
    Eout = logistic_error(Xout, Yout, weights)
    #display(Xout, Yout, weights, a, b)

    return Eout, iters_to_converge


nu_coef = 0.01
epsilon = 0.01
Nin = 100
Nout = 100


Eout = compute_logistic_error(Nin, Nout, nu_coef, epsilon)
print Eout

Eout_aver = 0
iters_aver = 0
Niter = 100
for i in range(Niter):
    Eout, iters_to_converge = compute_logistic_error(Nin, Nout, nu_coef, epsilon)
    iters_aver += iters_to_converge / Niter
    Eout_aver += Eout / Niter

print iters_aver
