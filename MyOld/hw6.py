import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm, inv


sign = lambda x: math.copysign(1, x)


def read(filename):
    data_file = open(filename, 'r')
    X, Y = [], []
    for line in data_file:
        line_items = line.split('  ')
        x_part = float(line_items[1]), float(line_items[2])
        X.append(x_part)
        Y.append(float(line_items[-1]))
    data_file.close()
    if len(X) != len(Y):
        raise ValueError
    return np.array(X), np.array(Y)


def short_transform(pair):
    x1, x2 = pair
    return 1, x1, x2, x1 ** 2, x2 ** 2, x1 * x2, abs(x1 - x2), abs(x1 + x2)


def transform(X_data):
    # data is a list of (x1, x2) pairs
    Z = []
    for pair in X_data:
        Z.append(short_transform(pair))
    return np.array(Z)


def compute_linreg_error(W, X, Y):
    N = X.shape[0]
    num_of_misclassified = 0.
    for i in range(N):
        binary_val = sign(W.T.dot(X[i].T))
        if binary_val * Y[i] < 0:
            num_of_misclassified += 1
    return float(num_of_misclassified / N)


def obtain_liner_weights(X, Y, lambda_reg=0):
    vc_dim = X.shape[1]
    I = np.eye(vc_dim)
    return (inv((X.T.dot(X) + lambda_reg * I)).dot(X.T)).dot(Y)


def get_the_best_lambda_rate(show=True):
    Xin, Yin = read('in.dta.txt')
    Xout, Yout = read('out.dta.txt')
    Zin = transform(Xin)
    Zout = transform(Xout)

    weights = obtain_liner_weights(Zin, Yin)
    Eout = compute_linreg_error(weights, Zout, Yout)

    Eout_min = Eout
    for k in range(-10, 10):
        lambda_reg = 10 ** k
        weights_reg = obtain_liner_weights(Zin, Yin, lambda_reg)
        Eout_reg = compute_linreg_error(weights_reg, Zout, Yout)
        plt.plot(k, Eout_reg, 'ro', hold=True)
        if Eout_reg < Eout_min:
            Eout_min = Eout_reg
    if show:
        plt.show()
    return Eout_min


def compute_weights_dimension(hidden_layers_dim):
    input_size = 10
    output_size = 2
    last_hidden_layer_size = 36 - sum(hidden_layers_dim)
    layers_dim = [input_size] + hidden_layers_dim + [last_hidden_layer_size, output_size]
    print layers_dim

    L = len(layers_dim)
    wsize = 0
    for l in range(1, L):
        wsize += layers_dim[l-1] * (layers_dim[l] - 1)
    return wsize



Xin, Yin = read('in.dta.txt')
Xout, Yout = read('out.dta.txt')

Zin = transform(Xin)
Zout = transform(Xout)

wsize = compute_weights_dimension([22])

print "\nweights dimension size:\t", wsize



