# encoding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pprint import pprint


def q1():
    d_vc = 10
    eps = 0.05
    p = lambda N: 4 * (2 * N)**d_vc * np.exp(-0.125 * eps**2 * N)
    n_list = [400000, 420000, 440000, 460000, 480000]
    probs = []
    for n in n_list:
        probs.append(p(n))
    probs = np.array(probs)
    diff = np.abs(probs - 0.05)
    n_best = n_list[np.argmin(diff)]
    print "Closest N = {}".format(n_best)


def c_n_k(n, k):
    res = 1
    for x in range(n-k+1, n+1):
        res *= x
    for x in range(1, k+1):
        res /= x
    return res


def m_n(n, d_vc):
    m = 0
    for d in range(d_vc+1):
        m += c_n_k(n, d)
    return m


def find_eps_upper(eps_func, n):
    eps_eval = []
    for eps in np.linspace(0, 100, num=50):
        eps_lower = eps_func(n, eps)
        if eps_lower <= eps:
            eps_eval.append(eps_lower)
    return max(eps_eval)


def q2():
    """
     Incorrect. Need to fix find_eps_upper func.
    """
    d_vc = 50
    p = 0.05
    eps_orig = lambda n: np.sqrt(8. / n * np.log(4. * m_n(2*n, d_vc) / p))
    eps_rademacher = lambda n: np.sqrt(2. * np.log(2. * n * m_n(n, d_vc)) / n) + np.sqrt(2./n * np.log(1./p)) + 1./n
    eps_parrondo = lambda n, eps: np.sqrt(1./n * (2. * eps) + np.log(6 * m_n(2*n, d_vc) / p))
    eps_devroye = lambda n, eps: np.sqrt(1/(2*n) * (4 * eps * (1 + eps)) + np.log(4 * m_n(n**2, d_vc) / p))
    for n in range(1, 5*int(1e3+1), int(1e2)):
        plt.plot(n, eps_orig(n), 'ro')
        plt.plot(n, eps_rademacher(n), 'bo')
        plt.plot(n, find_eps_upper(eps_parrondo, n), 'go')
        plt.plot(n, find_eps_upper(eps_devroye, n), 'rs')
    plt.legend(["orig", "rademarcher", "parrondo", "devroye"])
    plt.show()


def q456():
    n_iters = int(1e6)
    target_func = lambda x: np.sin(np.pi * x)

    def generate_dataset_with_coefs():
        """
        :return: (n_iters, 2) random points with approximated (n_iters,) coefficients
        """
        xs = 2. * np.random.random_sample(size=(n_iters, 2)) - 1.
        ys = target_func(xs)
        a_coef = np.sum(xs * ys, axis=1) / np.sum(xs ** 2, axis=1)
        return xs, a_coef

    xs, a_coef = generate_dataset_with_coefs()
    a_aver = np.mean(a_coef)

    xs_linspace = np.linspace(-1, 1, num=n_iters)
    bias = np.mean((a_aver * xs_linspace - target_func(xs_linspace)) ** 2)

    var = 0.
    for point_id in range(xs.shape[1]):
        var += np.sum((a_aver * xs[:, point_id] - a_coef * xs[:, point_id]) ** 2)
    var /= (xs.shape[0] * xs.shape[1])
    print "mean: {}".format(a_aver)
    print "bias: {}".format(bias)
    print "variance: {}".format(var)


def q7():
    n_iters = 1000
    n_out = 1000
    target_func = lambda x: np.sin(np.pi * x)
    xs_linspace = np.linspace(-1, 1, n_out)
    ys_linspace = target_func(xs_linspace)

    hypothesis = {
        "b": lambda x, b: b,
        "ax": lambda x, a: a * x,
        "ax + b": lambda x, a, b: a * x + b,
        "ax**2": lambda x, a: a * x**2,
        "ax**2 + b": lambda x, a, b: a * x**2 + b
    }

    e_out = {}
    for h_name, h_func in hypothesis.iteritems():
        h_error = 0.
        for i in range(n_iters):
            xs = 2. * np.random.random_sample(size=(2,)) - 1.
            ys = target_func(xs)
            popt, pcov = curve_fit(h_func, xs, ys)
            h_error += np.mean((h_func(xs_linspace, *popt) - ys_linspace) ** 2)
        h_error /= n_iters
        e_out[h_name] = h_error
    pprint(e_out)


def q8():
    """
    Prove that dvc == q, using Pascal's triangle.
    """
    q = np.random.randint(1, 10)
    n = np.random.randint(1, 100)
    for dvc in [q-2, q-1, q, q+1]:
        if m_n(n+1, dvc) == 2 * m_n(n, dvc) - c_n_k(n, q):
            print q, dvc

if __name__ == "__main__":
    q8()
