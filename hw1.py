import numpy as np
import matplotlib.pyplot as plt

sign = lambda x: 2 * (x > 0) - 1  # sign(0) = -1
y_func = lambda x, w: -w[0] / w[2] - w[1] / w[2] * x


class Perceptron(object):
    """ Perceptron always has 2 possible outputs (k = 2)"""

    def __init__(self, x_data):
        """
        :param x_data: (N, #features) - 2dim array of input data
        """
        n_samples = x_data.shape[0]
        features = x_data.shape[1]
        self.x_data = np.empty(shape=(n_samples, features + 1))
        self.x_data[:, 0] = 1  # added bias terms
        self.x_data[:, 1:] = x_data.copy()
        self.weights = np.zeros(features + 1)
        self.iterations = 0
        # self.weights = np.random.rand(features + 1)

    def forward(self):
        return np.array(sign(self.weights.dot(self.x_data.T)))

    def update_weights(self, y_train):
        hypothesis = self.forward()
        missed = hypothesis != y_train
        if missed.any():
            pid = np.random.choice(np.where(missed)[0])
            self.weights += y_train[pid] * self.x_data[pid, :]
        self.iterations += 1
        return missed.any()


def display(x_data, y_sign, weights_obt, a_target, b_target):
    """ Display data. """
    # plotting target line
    # plt.axis([-1, 1, -1, 1])
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


def run_demo():
    n_samples = 10
    a, b = get_rand_coefficients()

    x_data = 2 * np.random.random_sample(size=(n_samples, 2)) - 1
    y_sign = sign(x_data[:, 1] - (a * x_data[:, 0] + b))

    perceptron = Perceptron(x_data)
    while perceptron.update_weights(y_sign):
        # note: some data might not converge at all
        pass

    display(x_data, y_sign, perceptron.weights, a, b)


def is_inside_rect(x, y):
    return -1 <= x <= 1 and -1 <= y <= 1


def clamp(x, lower, upper):
    if x < lower:
        return lower
    elif x > upper:
        return upper
    else:
        return x

X_DATA = 0
Y_DATA = 0
WEIGHTS = 0
x_intersection = 0
y_intersection = 0
a_target = 0
b_target = 0


def calc_area(x0, x1, a, b):
    y0 = clamp(x0 * a + b, -1, 1)
    y1 = clamp(x1 * a + b, -1, 1)
    area = 0.0
    if y0 == 1.0:
        x0_top = (y0 - b) / a
        x_data = np.empty(shape=(4, 2 + 1))
        x_data[:, 0] = 1  # added bias terms
        x_data[:, 1:] = X_DATA.copy()
        if not (-1.0 < x0_top < x1):
            display(X_DATA, Y_DATA, WEIGHTS, a_target, b_target)
        assert -1.0 < x0_top < x1
        area += 2 * (x0_top - (-1))
        x0 = x0_top
    if y1 == 1.0:
        x1_top = (y1 - b) / a
        if not (x0 < x1_top < 1.0):
            display(X_DATA, Y_DATA, WEIGHTS, a_target, b_target)
        assert x0 < x1_top < 1.0
        area += 2 * (1.0 - x1_top)
        x1 = x1_top
    h0 = y0 - (-1)
    h1 = y1 - (-1)
    area += (h0 + h1) / 2 * (x1 - x0)
    return area


def calc_area_diff(x0, x1, a1, b1, a2, b2):
    s1 = calc_area(x0, x1, a1, b1)
    s2 = calc_area(x0, x1, a2, b2)
    return abs(s1 - s2)


def calc_error_out(a1, b1, w):
    global x_intersection, y_intersection
    a2 = -w[1] / w[2]
    b2 = -w[0] / w[2]
    x_intersection = (b2 - b1) / (a1 - a2)
    error_area = 0.0
    if is_inside_rect(x_intersection, y_intersection):
        error_area += calc_area_diff(-1, x_intersection, a1, b1, a2, b2)
        error_area += calc_area_diff(x_intersection, 1, a1, b1, a2, b2)
    else:
        error_area += calc_area_diff(-1, 1, a1, b1, a2, b2)
        error_area += calc_area_diff(-1, 1, a1, b1, a2, b2)
    return error_area / 4.0


def run_iterations(n_samples, iters):
    global X_DATA, Y_DATA, WEIGHTS, a_target, b_target
    aver_iters_to_converge = 0
    aver_error_out = 0.0
    for iter_id in range(iters):
        a, b = get_rand_coefficients()
        x_data = 2 * np.random.random_sample(size=(n_samples, 2)) - 1
        y_sign = sign(x_data[:, 1] - (a * x_data[:, 0] + b))
        if np.all(y_sign == 1) or np.all(y_sign == 0):
            continue
        # while np.any(np.array(y_sign) == -1):
        #     x_data = 2 * np.random.random_sample(size=(n_samples, 2)) - 1
        #     y_sign = sign(x_data[:, 1] - (a * x_data[:, 0] + b))
        perceptron = Perceptron(x_data)
        while perceptron.update_weights(y_sign):
            # note: some data might not converge at all
            pass
        X_DATA = x_data
        Y_DATA = y_sign
        WEIGHTS = perceptron.weights
        a_target = a
        b_target = b
        # display(x_data, y_sign, perceptron.weights, a, b)
        aver_iters_to_converge += perceptron.iterations
        aver_error_out += calc_error_out(a, b, perceptron.weights)
    aver_iters_to_converge /= float(iters)
    aver_error_out /= float(iters)
    print("Average iterations made to converge: %d" % aver_iters_to_converge)
    print("Average out-of-sample error: %f" % aver_error_out)


if __name__ == "__main__":
    run_iterations(4, 1000)
