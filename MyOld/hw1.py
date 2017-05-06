import random, math
import numpy as np
import matplotlib.pyplot as plt

sign = lambda x: math.copysign(1, x)

def display():
    """ Display all. """
    x0 = X[:, 1]    # as simple x in plot

    # plotting target line (as a classifier)
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


def get_rand_coef():
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


def classify(W, X):
    h = []
    for i in range(len(Y)):
        trigger = sign(W.dot(X[i]))
        h.append(trigger)
    return h


def upd_weights(X, h, Y, w):
    """ Returns random misclassified point from a misclassified set. """
    shit_list = []
    for i in range(len(Y)):
        if h[i] != Y[i]:
            shit_list.append([np.array(X[i]), Y[i]])
    if shit_list:
        x_rand, y_rand = random.choice(shit_list)
        return True, w + y_rand * x_rand
    else:
        return False, w


def get_divergence_prob(w, a, b):
    divergence_prob = 0
    sample_size = 1000
    inside = 0

    for i in range(sample_size):
        x0 = random.randrange(-100, 100, 1) / 100.0
        y_target = a * x0 + b
        if abs(w[2]) < 10 ** -3:
            w[2] = 10 ** -3
        y_obt = -w[0]/w[2] - w[1]/w[2] * x0

        y_rand = random.randrange(-100, 100, 1) / 100.0
        delta_y_target = y_target - y_rand
        delta_y_obt = y_target - y_obt

        if sign(delta_y_target) * sign(delta_y_obt) < 0:
            inside += 1
    return float(inside) / sample_size


N = 100
iterations_need = 0
aver_divergence_prob = 0

num_of_iterations = 1000
for dummy_iterator in range(num_of_iterations):
    X = []
    Y = []
    a, b = get_rand_coef()

    for i in range(N):
        x0 = random.randrange(-100, 100, 1) / 100.0
        x1 = random.randrange(-100, 100, 1) / 100.0
        X.append([1.0, x0, x1])
        delta = x1 - (a * x0 + b)
        Y.append(sign(delta))
    X = np.array(X)
    w = np.zeros(3).T

    iterator = 0
    shit_happens = True
    while shit_happens:
        iterator += 1
        h = classify(w, X)
        shit_happens, w = upd_weights(X, h, Y, w)
    # display()

    divergence_prob = get_divergence_prob(w, a, b)
    aver_divergence_prob += divergence_prob / num_of_iterations

    iterations_need += (iterator - 1) / float(num_of_iterations)

print(iterations_need, aver_divergence_prob)