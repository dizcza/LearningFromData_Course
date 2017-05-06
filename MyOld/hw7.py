import math
import matplotlib.pyplot as plt
import numpy as np
import random
from numpy.linalg import norm, inv
from numpy import sqrt

from cvxopt import matrix
from cvxopt import solvers

sign = lambda x: math.copysign(1, x)


def read(filename, split=False, Ntrain=25):
    """ Reads in.dta or out.dta with possibility to split into train and cross-valid datasets. """
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
    X, Y = np.array(X), np.array(Y)
    if split:
        Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
        Xval, Yval = X[Ntrain:], Y[Ntrain:]
        return Xtrain, Ytrain, Xval, Yval
    else:
        return X, Y


def short_transform(pair):
    """ Transforms a point from X --> a point from Z-space. """
    x1, x2 = pair
    return 1, x1, x2, x1 ** 2, x2 ** 2, x1 * x2, abs(x1 - x2), abs(x1 + x2)


def transform(X_data, k=8):
    """ Transforms X-data - vector of (x1, x2) pairs - into Z-space vector
        of dimensionality d = bias + 7. """
    Z = []
    for pair in X_data:
        Z.append(short_transform(pair)[:k])
    return np.array(Z)


def compute_error(W, X, Y):
    """ Compute the binary error for PLA.
        Y should be: Y = W * X. """
    N = X.shape[0]
    num_of_misclassified = 0.
    for i in range(N):
        binary_val = sign(W.T.dot(X[i].T))
        if binary_val * Y[i] < 0:
            num_of_misclassified += 1
    return float(num_of_misclassified / N)


def compute_weights(X, Y, lambda_reg=0):
    vc_dim = X.shape[1]
    I = np.eye(vc_dim)
    return (inv((X.T.dot(X) + lambda_reg * I)).dot(X.T)).dot(Y)


def disp_error_through_different_models(reverse=False, show_out_of_sample_error=False):
    """ Evaluates the out-of-sample performance by using the
        k-dim subspace of Z-space transformation. """
    if not reverse:
        Xtrain, Ytrain, Xval, Yval = read('in.dta.txt', split=True)
    else:
        Xval, Yval, Xtrain, Ytrain = read('in.dta.txt', split=True)
    Xout, Yout = read('out.dta.txt')
    weights = {}
    Eval = {}
    for k in range(3, 8):
        Ztrain = transform(Xtrain, k+1)
        Zval = transform(Xval, k+1)
        weights[k] = compute_weights(Ztrain, Ytrain)
        Eval[k] = compute_error(weights[k], Zval, Yval)
        model_status = "K = %i: Eval = %.2f" % (k, Eval[k])
        if show_out_of_sample_error:
            Zout = transform(Xout, k+1)
            Eout = compute_error(weights[k], Zout, Yout)
            model_status += " Eout = %.2f" % Eout
        print model_status


def vectorize(point1, point2):
    """ Returns a vector from a point1 to a point2. """
    return float(point2[0] - point1[0]), float(point2[1] - point1[1])


def fit_linear_model(points):
    """ Fits the 2 points using the linear regression:
        y = a*x + b.
        points: [(x0, y0), (x1, y1)]
    """
    x0, y0 = points[0]
    dx, dy = vectorize(points[0], points[1])
    a = dy/dx
    b = y0 - a * x0
    return a, b


def fit_const_model(points):
    """ Fits the 2 points using the linear const regression:
        y = a - const
        points: [(x0, y0), (x1, y1)]
    """
    pX, pY = zip(*points)
    return float(sum(pY)) / len(pY)


def quiz7():
    """ Evaluates the cross-validation performance for the data of three points. """
    Eval = {'constant': [],
            'linear': []}
    param_values = [sqrt(sqrt(3) + 4),
                    sqrt(sqrt(3) - 1),
                    sqrt(9 + 4 * sqrt(6)),
                    sqrt(9 - sqrt(6))]
    for param in param_values:
        points = (-1, 0), (param, 1), (1, 0)
        Eval_linear = 0.0
        Eval_const = 0.0
        for pid in range(3):
            x_out, y_out = points[pid]
            training_points = [points[i] for i in range(3) if i != pid]
            a, b = fit_linear_model(training_points)
            constant = fit_const_model(training_points)
            Eval_linear += abs(a * x_out + b - y_out)
            Eval_const += abs(constant - y_out)
        Eval['linear'].append(Eval_linear)
        Eval['constant'].append(Eval_const)
    return Eval


# ----------------------------------------------------------------------------
def get_rand_coef():
    """ Returns random coef a and b from the X-space: [-1, 1]x[-1, 1]. """
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


def unwrap_weights(weights, X_axis):
    """ weights = [w0, w1, w2] <--> w0 + w1 * x1 + w2 * x2 = 0
        Unwraps the X_axis (which is simply x1) into the obtained classified line:
        x2 = -w0/w2 - w1/w2 * x1
    """
    if abs(weights[2]) < 10 ** -2:
        weights[2] = 10 ** -2
    Y_obt = -weights[0]/weights[2] - weights[1]/weights[2] * X_axis
    return Y_obt


def display(X, Y, w, a, b, weights_svm=None, show=True):
    """ Display all points in X = [-1, 1]x[-1, 1]. """
    N = X.shape[0]
    X_axis = X[:, 1]    # as simple x in plot

    # plotting target line (hidden function)
    x_target = np.array(range(-100, 101)) / 100.0
    y_target = a * x_target + b
    plt.axis([-1, 1, -1, 1])
    plt.plot(x_target, y_target, '--', lw=1)

    # plotting obtained linear classifier
    Y_obt = unwrap_weights(w, X_axis)
    plt.plot(X_axis, Y_obt, lw=1, color='black')
    legend = ['hidden function', 'obtained classifier']
    if weights_svm.any():
        Y_obt = unwrap_weights(weights_svm, X_axis)
        plt.plot(X_axis, Y_obt, lw=2, color='green')
        legend.append(['SVM optimized'])
    plt.legend(legend, loc=0)

    # plotting labeled points
    for i in range(N):
        if Y[i] < 0:
            plt.plot(X[i][1], X[i][2], 'ro')
        else:
            plt.plot(X[i][1], X[i][2], 'go')
    if show:
        plt.show()


def generate_data(N, a, b):
    """
        Generates a binary data:
            Y = {-1, 1}
            X = [-1, 1] x [-1, 1].
            N - number of points to generate
            a, b - hidden coef from the target func
    """
    while True:
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
        if abs(sum(Y)) < len(Y):
            # checks whether the Y includes both of -1 and +1 labels
            # and returns X and Y, if so
            return X, Y


def upd_weights(weights, Xtr, Ytr, Y_obtained):
    """ Randomly picks one misclassified point from a misclassified set and returns it. """
    shit_list = []
    for i in range(len(Ytr)):
        if Y_obtained[i] != Ytr[i]:
            shit_list.append([np.array(Xtr[i]), Ytr[i]])
    if shit_list:
        x_rand, y_rand = random.choice(shit_list)
        return True, weights + y_rand * x_rand
    else:
        return False, weights


def classify(W, X, Y):
    """ Returns the predicted labels for all X, w.r. to weights W """
    Y_obtained = []
    for i in range(len(Y)):
        trigger = sign(W.dot(X[i]))
        Y_obtained.append(trigger)
    return Y_obtained
# ---------------------------------------------------------------------

def find_closest_point_id(X, weights):
    """ Finds the closest point to the separating line. """
    w0, w1, w2 = weights
    dists = []
    for point in X:
        _, x1, x2 = point
        x2_obtained = -w1/w2 * x1 - w0 / w2
        dist_to_line = abs((x2 - x2_obtained) * w2)
        dists.append(dist_to_line)
    return np.argmin(dists)


def build_Q(X, Y):
    """ Builds the quadratic matrix Q for the programming task solver. """
    N = X.shape[0]
    Q = matrix(0.0, (N, N))
    for col in range(N):
        for row in range(N):
            Q[col, row] = Y[col] * Y[row] * X[col].dot(X[row])
    return Q


def get_solution(Xtr, Ytr):
    """ Returns the cvxopt solver, pertains to the data X and Y. """
    solvers.options['show_progress'] = False
    Nin = len(Ytr)
    P = build_Q(Xtr, Ytr)
    q = matrix(-1.0, (Nin, 1))
    G = - matrix(np.eye(Nin))
    h = matrix(0.0, (Nin, 1))
    A = matrix(Ytr, (1, Nin))
    b = matrix(0.0, (1, 1))
    return solvers.qp(P, q, G, h, A, b)


def run_svm_optimization(Xtr, Ytr, alphas, non_zero_threshold):
    """ Runs the optimization technique.
        Returns:
            the weights_margin = [b, w1, w2]
            support vectors x_n with corresponding labels y_n
    """
    non_zero_alphas = [non_zero_val for non_zero_val in alphas if non_zero_val > non_zero_threshold]
    Nin = len(Ytr)
    weights_svm = np.zeros(2)
    support_vectors = []
    support_vectors_labels = []
    for pid in range(Nin):
        if alphas[pid] > non_zero_threshold:
            weights_svm += alphas[pid] * Ytr[pid] * Xtr[pid][1:]
            support_vectors.append(list(Xtr[pid]))
            support_vectors_labels.append(Ytr[pid])
    support_vectors = np.array(support_vectors)
    support_vectors_labels = np.array(support_vectors_labels)

    the_same_w0_val = []
    for spid in range(len(non_zero_alphas)):
        w0 = 1.0 / support_vectors_labels[spid] - weights_svm.dot(support_vectors[spid][1:])
        the_same_w0_val.append(w0)
    if np.std(the_same_w0_val) / abs(np.mean(the_same_w0_val)) > 0.01:
        print "The w0 values aren't the same: ", the_same_w0_val
    weights_margin = np.append(the_same_w0_val[0], weights_svm)

    return weights_margin, support_vectors, support_vectors_labels


def quiz8(Nin=10, Nout=100):
    """ Main function. """
    # step 0: generate datasets
    a, b = get_rand_coef()
    Xtr, Ytr = generate_data(Nin, a, b)

    # step 1: find weights, using PLA
    PLA_step = 0
    shit_happens = True
    weights = np.zeros(Xtr.shape[1])
    while shit_happens:
        PLA_step += 1
        Y_obtained = classify(weights, Xtr, Ytr)
        shit_happens, weights = upd_weights(weights, Xtr, Ytr, Y_obtained)

    # step 2: find the closest point to the final hypothesis g
    closest_point_id = find_closest_point_id(Xtr, weights)
    closest_point = Xtr[closest_point_id]

    # show the closest point
    # plt.plot(closest_point[1], closest_point[2], 'bo', markersize=10)

    # step 3: normalize weights
    weights_pla = weights / weights.dot(closest_point)
    if weights_pla.dot(closest_point) < 1.0 - 10 ** -3:
        print "weights_pla.dot(closest_point): ", weights_pla.dot(closest_point)

    # step 4: find distance to the hyperplane
    dist = 1.0 / np.linalg.norm(weights_pla[1:])

    # step 7: solve quadratic programming task and find non-zero alphas
    sol = get_solution(Xtr, Ytr)
    alphas = sol['x']
    non_zero_threshold = 10.0 ** -4
    non_zero_alphas = [non_zero_val for non_zero_val in alphas if non_zero_val > non_zero_threshold]
    #return len(non_zero_alphas)

    print "Non-zero alphas: %s, length = %i" % (non_zero_alphas, len(non_zero_alphas))

    # step 8: find SVs and compute weights, which yield the biggest margin
    weights_svm_full, \
    support_vectors, \
    support_vectors_labels = run_svm_optimization(Xtr,
                                                  Ytr,
                                                  alphas,
                                                  non_zero_threshold)
    # step 9: PLA vs SVM results
    Xtest, Ytest = generate_data(Nout, a, b)
    Eout_pla = compute_error(weights, Xtest, Ytest)
    Eout_svm = compute_error(weights_svm_full, Xtest, Ytest)
    if Eout_pla == 0.0:
        how_better = 0.0
    else:
        how_better = (Eout_pla - Eout_svm) / Eout_pla

    if how_better < 0 and 0:
        plt.figure(1)
        display(Xtr, Ytr, weights, a, b, weights_svm_full, show=1)
        plt.figure(2)
        display(Xtest, Ytest, weights, a, b, weights_svm_full, show=1)
        #plt.show()
    return Eout_svm < Eout_pla


"""better_perf_aver = 0.0
n_support_vectors = 0.0

Niter = 1000
for dummy_i in range(Niter):
    non_zero_alphas = quiz8(Nin=100, Nout=10000)
    n_support_vectors += non_zero_alphas / float(Niter)

print "\nBETTER PERFORMANCE: %g" % n_support_vectors
"""

