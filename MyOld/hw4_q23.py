from math import factorial, exp, log, sqrt, sin
import matplotlib.pyplot as plt


def m(d_vc, N):
    """ The growth function. """
    if N < d_vc:
        return 2 ** N
    else:
        return N ** d_vc


def original_bound(d_vc, N, delta, eps=0):
    return sqrt(8/N * log(4 * m(d_vc, 2*N)/delta))

def rademacher_bound(d_vc, N, delta, eps=0):
    return sqrt(2 * log(2*N * m(d_vc, N))/N) + sqrt(log(1/delta) / N) + 1/N

def parrondo_bound(d_vc, N, delta, eps):
    return sqrt((2 * eps + log(6 * m(d_vc, 2*N)/delta)) / N)

def devroye_bound(d_vc, N, delta, eps):
    ln_val = log(4 * m(d_vc, N**2) / delta)
    sigma = (4 * eps * (1 + eps) + ln_val) / (2 * N)
    return sqrt(sigma)

def quizzes2to3_show():
    bound_functions = [original_bound, rademacher_bound, parrondo_bound, devroye_bound]
    delta = 0.05
    d_vc = 50
    N = 1000.0

    for get_bound in bound_functions:
        X_epsilons = []
        Y_eps_bound = []
        for epsilon in range(100):
            eps = epsilon / 100.0
            X_epsilons.append(eps)
            eps_upper = get_bound(d_vc, N, delta, eps)
            Y_eps_bound.append(eps_upper)
        plt.plot(X_epsilons, Y_eps_bound)
        #print Y_eps_bound

    plt.legend(["original_bound", "rademacher_bound", "parrondo_bound", "devroye_bound"])
    plt.xlabel("epsilon")
    plt.ylabel("epsilon_upper_bound")
    plt.show()

quizzes2to3_show()
