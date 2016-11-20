# encoding=utf-8

import numpy as np
import matplotlib.pyplot as plt

sign = lambda x: 2 * (x > 0) - 1  # sign(0) = -1
y_func = lambda x, w: -w[0] / w[2] - w[1] / w[2] * x

class LinearRegression(object):
    def __init__(self, x_data, y_sign):
        """
        :param x_data: (N, #features) - 2dim array of input data
        """
        features = x_data.shape[1]
        self.x_data = add_bias_term(x_data)
        self.y_sign = y_sign
        self.weights = np.zeros(features + 1)
        self.iterations = 0

    def calc_weights(self):
        x_pseudo_inv = np.linalg.inv(np.dot(self.x_data.T, self.x_data)).dot(self.x_data.T)
        self.weights = x_pseudo_inv.dot(self.y_sign)
        return self.weights

    def get_in_sample_error(self):
        y_predicted = sign(self.x_data.dot(self.weights))
        error_in = np.average(y_predicted != self.y_sign)
        return error_in

class PerceptronFromLinearRegression(object):
    """ Perceptron always has 2 possible outputs (k = 2)"""

    def __init__(self, x_data, y_sign, weights_initial):
        """
        :param x_data: (N, #features) - 2dim array of input data
        """
        self.x_data = add_bias_term(x_data)
        self.y_sign = y_sign
        self.weights = weights_initial
        self.iterations = 0

    def forward(self):
        return np.array(sign(self.weights.dot(self.x_data.T)))

    def update_weights(self):
        hypothesis = self.forward()
        missed = hypothesis != self.y_sign
        if missed.any():
            pid = np.random.choice(np.where(missed)[0])
            self.weights += self.y_sign[pid] * self.x_data[pid, :]
        self.iterations += 1
        return missed.any()

def display(x_data, y_sign, weights_obt, a_target, b_target):
    """ Display data. """
    # plotting target line
    plt.axis([-1, 1, -1, 1])
    x_borders = np.array([-1, 1])
    y_borders = a_target * x_borders + b_target
    plt.plot(x_borders, y_borders, '--', lw=3, label="target")

    # plotting obtained linear classifier
    y_obt = y_func(x_borders, weights_obt)
    plt.plot(x_borders, y_obt, lw=1, color='black', label="obtained")
    plt.legend()

    # plotting labeled points
    for xp, label in zip(x_data, y_sign):
        plt.plot(xp[0], xp[1], 'ro' if label < 0 else 'go')

    plt.show()

def add_bias_term(x_arr):
    x_biased = np.empty(shape=(x_arr.shape[0], x_arr.shape[1] + 1))
    x_biased[:, 0] = 1  # added bias terms
    x_biased[:, 1:] = x_arr.copy()
    return x_biased

def get_rand_coefficients():
    p0, p1 = 2 * np.random.random_sample(size=(2, 2)) - 1
    a = (p1[1] - p0[1]) / (p1[0] - p0[0])
    b = p0[1] - a * p0[0]
    ys = []
    for x in np.linspace(-1.0, 1.0, num=10000):
        y = a * x + b
        if -1.0 <= y <= 1.0:
            ys.append(y)
    assert len(ys) > 0
    return a, b

def create_linear_data_set(n_samples, a, b):
    x_data = 2 * np.random.random_sample(size=(n_samples, 2)) - 1
    y_sign = sign(x_data[:, 1] - (a * x_data[:, 0] + b))
    return x_data, y_sign

def calc_error_out(a, b, w, n_out_samples=1000):
    x_data, y_sign_truth = create_linear_data_set(n_out_samples, a, b)
    x_biased = add_bias_term(x_data)
    y_sign_obt = sign(w.dot(x_biased.T))
    e_out = np.average(y_sign_obt != y_sign_truth)
    return e_out

def run_linear_regression_as_perceptron(n_samples, n_iters=1000):
    linear_regr_error_in = 0.0
    linear_regr_error_out = 0.0
    perceptron_iters_to_converge = 0
    for iter_id in range(n_iters):
        a, b = get_rand_coefficients()
        x_data, y_sign = create_linear_data_set(n_samples, a, b)

        liner_regr = LinearRegression(x_data, y_sign)
        liner_regr.calc_weights()
        # display(x_data, y_sign, liner_regr.weights, a, b)
        linear_regr_error_in += liner_regr.get_in_sample_error()
        linear_regr_error_out += calc_error_out(a, b, liner_regr.weights)

        perceptron = PerceptronFromLinearRegression(x_data, y_sign, liner_regr.weights)
        while perceptron.update_weights():
            # we assume the data is linearly separable
            pass
        perceptron_iters_to_converge += perceptron.iterations

    linear_regr_error_in /= float(n_iters)
    linear_regr_error_out /= float(n_iters)
    perceptron_iters_to_converge /= float(n_iters)
    print("Average in-sample error: %f" % linear_regr_error_in)
    print("Average out-of-sample error: %f" % linear_regr_error_out)
    print("Average iterations to converge from LinearRegression: %d" % perceptron_iters_to_converge)

if __name__ == "__main__":
    run_linear_regression_as_perceptron(n_samples=10)

