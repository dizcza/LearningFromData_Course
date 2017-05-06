import numpy as np
from numpy import copysign, sin, pi, exp
from numpy.linalg import inv
import matplotlib.pyplot as plt
import random

# in order to use svmutil, install libsvm package
from svmutil import *

from scipy.cluster.vq import kmeans, vq

sign = lambda x: copysign(1, x)


def read_one_versus_all(filename, digit_matter):
    """ Returns the output - X, Y - in the one-versus-all labeling format. """
    data_file = open(filename, 'r')
    X, Y = [], []
    for line in data_file:
        line_items = line.strip(' ').split('  ')
        digit_online = int(float(line_items[0]))

        symmetry = float(line_items[1])
        intensity = float(line_items[2])

        features = [symmetry, intensity]
        X.append(features)

        if digit_online == digit_matter:
            Y.append(1.0)
        else:
            Y.append(-1.0)
    data_file.close()
    return np.array(X), np.array(Y)


def read_one_versus_another(filename, first_digit, second_digit):
    """ Returns the output - X, Y - in the one-versus-another labeling format. """
    data_file = open(filename, 'r')
    X, Y = [], []
    for line in data_file:
        line_items = line.strip(' ').split('  ')
        digit_online = int(float(line_items[0]))

        symmetry = float(line_items[1])
        intensity = float(line_items[2])
        features = [symmetry, intensity]

        if digit_online == first_digit:
            Y.append(1.0)
            X.append(features)
        elif digit_online == second_digit:
            Y.append(-1.0)
            X.append(features)
    data_file.close()
    return np.array(X), np.array(Y)


def compute_linreg_error(weights, X, Y):
    N = X.shape[0]
    num_of_misclassified = 0.
    for i in range(N):
        binary_val = sign(weights.T.dot(X[i].T))
        if binary_val * Y[i] < 0:
            num_of_misclassified += 1
    return float(num_of_misclassified / N)


def obtain_linear_weights(X, Y, lambda_reg=.0):
    vc_dim = X.shape[1]
    I = np.eye(vc_dim)
    return (inv((X.T.dot(X) + lambda_reg * I)).dot(X.T)).dot(Y)


def transform(X_data):
    """
    :param X_data: list of (x1, x2) pairs
    :return: the same data in Z-space
    """
    Z = []
    for pair in X_data:
        x1, x2 = pair
        transformed_data = 1, x1, x2, x1 * x2, x1 ** 2, x2 ** 2
        Z.append(transformed_data)
    return np.array(Z)


def disp_min_and_max_error(error_dic, error_type=""):
    """ Pretty printing the error boundaries for quizzes 2 and 3. """
    Ein_min = 1.0
    digit_Emin = 0
    Ein_max = 0.0
    digit_Emax = 0
    for digit, error in error_dic.iteritems():
        if error < Ein_min:
            digit_Emin = digit
            Ein_min = error
        if error > Ein_max:
            digit_Emax = digit
            Ein_max = error
    print error_type
    print "\t min error = %f with corresponding digit %i" % (Ein_min, digit_Emin)
    print "\t max error = %f with corresponding digit %i" % (Ein_max, digit_Emax)


def run_one_versus_all(do_transform=None):
    insample_errors = {}
    outofsample_errors = {}
    decimal_digits = [digit for digit in range(10)]

    if do_transform == "quadratic":
        print "\n***Quadratic transform***"
    else:
        print "\n***(without any tranform)***"

    for digit_matter in [0, 5, 9]:
        Xtr, Ytr = read_one_versus_all('features.train.txt', digit_matter)
        Xtest, Ytest = read_one_versus_all('features.test.txt', digit_matter)

        if do_transform == "quadratic":
            Ztr = transform(Xtr)
            Ztest = transform(Xtest)
        else:
            Ztr = np.array(Xtr)
            Ztest = np.array(Xtest)

        weights = obtain_linear_weights(Ztr, Ytr, lambda_reg=1.0)

        Ein = compute_linreg_error(weights, Ztr, Ytr)
        Eout = compute_linreg_error(weights, Ztest, Ytest)

        insample_errors[digit_matter] = Ein
        outofsample_errors[digit_matter] = Eout

    disp_min_and_max_error(insample_errors, "Ein")
    disp_min_and_max_error(outofsample_errors, "Eout")


def run_one_versus_other(first_digit, second_digit, lambda_reg=1.0, do_transform="quadratic"):
    insample_errors = {}
    outofsample_errors = {}

    input_info = "'%d' versus '%d'" % (first_digit, second_digit)
    if do_transform == "quadratic":
        print "\n*** %s : quadratic transform; lambda : %f ***" % (input_info, lambda_reg)
    else:
        print "\n*** %s : (without any tranform); lambda: %f ***" % (input_info, lambda_reg)

    Xtr, Ytr = read_one_versus_another('features.train.txt', first_digit, second_digit)
    Xtest, Ytest = read_one_versus_another('features.test.txt', first_digit, second_digit)

    if do_transform == "quadratic":
        Ztr = transform(Xtr)
        Ztest = transform(Xtest)
    else:
        Ztr = np.array(Xtr)
        Ztest = np.array(Xtest)

    weights = obtain_linear_weights(Ztr, Ytr, lambda_reg)

    Ein = compute_linreg_error(weights, Ztr, Ytr)
    Eout = compute_linreg_error(weights, Ztest, Ytest)
    print "Ein: %f, \t Eout: %f" % (Ein, Eout)


def run_svm():
    X = np.array([1.,0., 0.,1., 0.,-1., -1.,0., 0.,2., 0.,-2., -2.,0.]).reshape(7,2)
    Y = np.array([-1., -1., -1., 1., 1., 1., 1.])

    Z1 = X[:,1] ** 2 - 2 * X[:,0] - 1.
    Z2 = X[:,0] ** 2 - 2 * X[:,1] + 1.
    for n in range(len(Y)):
        if Y[n] == -1.:
            plt.plot(Z1[n], Z2[n], 'ro')
        elif Y[n] == 1.:
            plt.plot(Z1[n], Z2[n], 'go')
        else:
            raise ValueError
    boundaries = min(Z1) - 1., max(Z1) + 1., min(Z2) - 1., max(Z2) + 1.
    plt.axis(boundaries)
    plt.show()

    print "Run SVM..."
    const_info = "-c inf -t 1 -d 2 -g 1 -r 1 -h 0 -q"
    param = svm_parameter(const_info)
    print param
    problem = svm_problem(Y, X.tolist())
    model = svm_train(problem, param)
    support_vectors = len(model.get_SV())
    accuracy_insample = svm_predict(Y, X.tolist(), model)[1][0]
    print "#SV: %d" % support_vectors


#################################################################
# --------- Random data [-1, 1]x[-1, 1] preprocessor -----------
#################################################################
def display(X, Y):
    """ Display all points in X = [-1, 1]x[-1, 1]. """
    N = X.shape[0]
    x0 = X[:, 1]    # as simple x in plot

    # plotting target line (hidden function)
    x1 = np.array(range(-100, 101)) / 100.0
    y_target = x1 - 0.25 * sin(pi * x1)
    plt.axis([-1.1, 1.1, -1.1, 1.1])
    plt.plot(x1, y_target, '--', lw=3)

    # plotting labeled points
    for i in range(N):
        if Y[i] == -1.0:
            plt.plot(X[i][1], X[i][2], 'ro')
        elif Y[i] == 1.0:
            plt.plot(X[i][1], X[i][2], 'go')
        else:
            raise ValueError
    plt.show()


def generate_data(N, add_bias=False):
    """
        Generates a binary data:
            Y = {-1, 1}
            X = [-1, 1] x [-1, 1].
        Target function:
            f(x) = sign(x2 - x1 + 0.25sin(pi*x1))
    """
    X, Y = [], []
    for i in range(N):
        x1 = random.randrange(-100, 100, 1) / 100.0
        x2 = random.randrange(-100, 100, 1) / 100.0
        if add_bias:
            X.append([1.0, x1, x2])
        else:
            X.append([x1, x2])
        Y.append(sign(x2 - x1 + 0.25 * sin(pi * x1)))
    return np.array(X), np.array(Y)


#################################################################
# ----------------- Radial Basis Functions ----------------------
#################################################################
def run_rbf_clustering(Nin=100, gamma=1.5, K=9, Xin=None, Yin=None):
    if Xin.any() and Yin.any():
        X_short = Xin
        Y = Yin
    else:
        X_short, Y = generate_data(Nin)

    # handling with empty-clusters
    centroids = kmeans(X_short, K)[0]
    while len(set(vq(X_short, centroids)[0])) < K:
        centroids = kmeans(X_short, K)[0]

    # computing F-matrix and weights
    F_matrix = []
    for n in range(Nin):
        bias_term = [1.0]
        row = bias_term + [exp(-gamma * (X_short[n] - centroids[k]).dot(X_short[n] - centroids[k])) for k in range(K)]
        F_matrix.append(row)
    F_matrix = np.array(F_matrix)
    weights_full = inv(F_matrix.T.dot(F_matrix)).dot(F_matrix.T).dot(Y)
    #w0, weights = weights_full[0], weights_full[1:]
    return weights_full, centroids


def compute_clust_error(X, Y, weights_full, gamma, centroids):
    w0, weights = weights_full[0], weights_full[1:]
    N = len(Y)
    K = len(centroids)
    Y_clust = []
    for n in range(N):
        signal = sum([weights[k] * exp(-gamma * (X[n] - centroids[k]).dot(X[n] - centroids[k])) for k in range(K)]) + w0
        Y_clust.append(sign(signal))
    return 1.0 - np.array(Y_clust).dot(Y)/N


def run_rbf_svmkernel(Nin=100, gamma=1.5, Xin=[], Yin=[]):
    if len(Xin) > 0 and len(Yin) > 0:
        X = Xin.tolist()
        Y = Yin
    else:
        X, Y = generate_data(Nin)
        X = X.tolist()

    # Run SVM...
    const_info = "-c inf -t 2 -g " + str(gamma) + " -h 0 -q"
    param = svm_parameter(const_info)
    problem = svm_problem(Y, X)
    svm_model = svm_train(problem, param)
    #print "#SV: %d" % len(svm_model.get_SV())

    # in case of being always separable (in terms of Ein = 0)
    return svm_model


def clustering_versus_svmkernel(Nin=100, Nout=100, gamma=1.5, K=9):
    Xtr, Ytr = generate_data(Nin)
    Xtest, Ytest = generate_data(Nout)

    # compute out-of-sample performance, using regular form of RBF
    weights_full, centroids = run_rbf_clustering(gamma=gamma, K=K, Xin=Xtr, Yin=Ytr)
    Eout_clust = compute_clust_error(Xtest, Ytest, weights_full, gamma, centroids)

    # compute out-of-sample performance, using svm gaussian kernel
    svm_model = run_rbf_svmkernel(gamma=gamma, Xin=Xtr, Yin=Ytr)
    accuracy_outsamp = svm_predict(Ytest, Xtest.tolist(), svm_model)[1][0]
    Eout_svm = 1.0 - accuracy_outsamp / 100.0

     # checks whether SVM model beats clustering model
    return Eout_svm < Eout_clust


# svm_won = 0.0
# Niter = 100
# for i in range(Niter):
#     svm_won += clustering_versus_svmkernel(K=3)
# print "SVM won in %g%% times" % (svm_won / Niter * 100.0)


def compare_num_of_centroids():
    """
        Compares in-sample and out-of-sample errors w.r.t. different num K of centroids
    """
    Nin = 100
    Nout = 1000
    Xtr, Ytr = generate_data(Nin)
    Xtest, Ytest = generate_data(Nout)
    Ein, Eout = [], []
    for num_clusters in [9, 12]:
        weights_full, centroids = run_rbf_clustering(gamma=1.5, K=num_clusters, Xin=Xtr, Yin=Ytr)
        Ein.append(compute_clust_error(Xtr, Ytr, weights_full, gamma=1.5, centroids=centroids))
        Eout.append(compute_clust_error(Xtest, Ytest, weights_full, gamma=1.5, centroids=centroids))
    return np.array(Ein), np.array(Eout)


def compare_gammas():
    """
        Compares in-sample and out-of-sample errors w.r.t. different gammas
    """
    Nin = 100
    Nout = 1000
    Xtr, Ytr = generate_data(Nin)
    Xtest, Ytest = generate_data(Nout)
    Ein, Eout = [], []
    for gamma_val in [1.5, 2.0]:
        weights_full, centroids_got = run_rbf_clustering(gamma=gamma_val, K=9, Xin=Xtr, Yin=Ytr)
        Ein.append(compute_clust_error(Xtr, Ytr, weights_full, gamma=gamma_val, centroids=centroids_got))
        Eout.append(compute_clust_error(Xtest, Ytest, weights_full, gamma=gamma_val, centroids=centroids_got))
    return np.array(Ein), np.array(Eout)


def how_often_insample_error_got_zero():
    Niter = 1000
    got_zero = .0
    for i in range(Niter):
        Xtr, Ytr = generate_data(N=100)
        weights_full, centroids_got = run_rbf_clustering(gamma=1.5, K=9, Xin=Xtr, Yin=Ytr)
        Ein = compute_clust_error(Xtr, Ytr, weights_full, gamma=1.5, centroids=centroids_got)
        if Ein < 1e-5:
            got_zero += 1.
    print "Ein got zero %g%% times." % (100.0 * got_zero / Niter)


"""
# it's for quizzes 16 and 17
Niter = 10
Ein, Eout = np.array([.0, .0]), np.array([.0, .0])
for i in range(Niter):
    e_in, e_out = compare_gammas()
    Ein += e_in / Niter
    Eout += e_out / Niter
    if i == Niter // 2:
        print "...half is done..."

print "Ein: %s" % Ein
print "Eout: %s" % Eout
"""

