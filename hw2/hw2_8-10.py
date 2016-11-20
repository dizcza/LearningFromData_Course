# encoding=utf-8

import numpy as np
from hw2_567 import sign, LinearRegression

def add_noise(y_orig, noise=0.1):
    y_noisy = np.empty(y_orig.shape)
    for i in range(len(y_orig)):
        y_noisy[i] = y_orig[i]
        flip_sign = np.random.uniform() < noise
        if flip_sign:
            y_noisy[i] *= -1
    return y_noisy

def transform_nonlinear(x_data):
    x_nonlinear = np.empty(shape=(x_data.shape[0], 6))
    x_nonlinear[:, 0] = 1  # added bias terms
    x_nonlinear[:, 1:3] = x_data.copy()
    x_nonlinear[:, 3] = x_data[:, 0] * x_data[:, 1]
    x_nonlinear[:, 4:] = x_data ** 2
    return x_nonlinear

def calc_error_out_nonlinear(w, n_out_samples=1000):
    x_data, y_sign_truth = create_nonlinear_data_set(n_out_samples)
    x_nonlinear = transform_nonlinear(x_data)
    y_sign_obt = sign(x_nonlinear.dot(w))
    e_out = np.average(y_sign_obt != y_sign_truth)
    return e_out

def create_nonlinear_data_set(n_samples):
    x_data = 2 * np.random.random_sample(size=(n_samples, 2)) - 1
    y_sign = sign(x_data[:, 0] ** 2 + x_data[:, 1] ** 2 - 0.6)
    y_sign = add_noise(y_sign, noise=0.1)
    return x_data, y_sign

def run_noisy_nonlinear_regression(n_samples, n_iters=1000):
    linear_regr_error_in = 0.0
    nonlinear_regr_error_in = 0.0
    nonlinear_regr_error_out = 0.0
    nonlinear_weights = np.zeros(6)
    for iter_id in range(n_iters):
        x_data, y_sign = create_nonlinear_data_set(n_samples)

        liner_regr = LinearRegression(x_data, y_sign)
        liner_regr.calc_weights()
        # display(x_data, y_sign, liner_regr.weights, a, b)
        linear_regr_error_in += liner_regr.get_in_sample_error()

        x_transformed_no_bias = transform_nonlinear(x_data)[:, 1:]
        nonliner_regr = LinearRegression(x_transformed_no_bias, y_sign)
        nonlinear_weights += nonliner_regr.calc_weights()
        nonlinear_regr_error_in += nonliner_regr.get_in_sample_error()
        nonlinear_regr_error_out += calc_error_out_nonlinear(nonliner_regr.weights)

    linear_regr_error_in /= float(n_iters)
    nonlinear_regr_error_in /= float(n_iters)
    nonlinear_regr_error_out /= float(n_iters)
    nonlinear_weights /= float(n_iters)
    print("Average LinearRegression in-sample error: %f" % linear_regr_error_in)
    print("Average NonLinearRegression in-sample error: %f" % nonlinear_regr_error_in)
    print("Average NonLinearRegression out-of-sample error: %f" % nonlinear_regr_error_out)
    print("Average NonLinearRegression weights [1, x1, x2, x1x2, x1**2, x2**2] : ", nonlinear_weights)

if __name__ == "__main__":
    run_noisy_nonlinear_regression(n_samples=1000)

