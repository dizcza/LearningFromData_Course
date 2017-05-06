import math
import matplotlib.pyplot as plt
import numpy as np

# in order to use svmutil, install libsvm package
from svmutil import *

sign = lambda x: math.copysign(1, x)


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
    return X, Y


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
    return X, Y


def disp_min_and_max_error(error_dic, error_type=""):
    """ Pretty printing the error boundaries for quizzes 2 and 3. """
    Ein_min = 100.0
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
    print "\n%s:" % error_type
    print "min error = %f%% with corresponding digit %i" % (Ein_min, digit_Emin)
    print "max error = %f%% with corresponding digit %i" % (Ein_max, digit_Emax)


def one_versus_all():
    """ One versus all implementation without the cross-validation.
        kernel: polynomial: (gamma * (u' .dot v) + coef) ^ degree
    """
    Ein = {}
    Eout = {}
    #quiz_digits = [1, 3, 5, 7, 9]
    quiz_digits = [i for i in range(10)]
    for digit in quiz_digits:
        Xtr, Ytr = read_one_versus_all('features.train.txt', digit)
        param = svm_parameter("-c 0.01 -t 1 -d 2 -g 1 -r 1 -h 0 -q")
        problem = svm_problem(Ytr, Xtr)
        model = svm_train(problem, param)

        labels_trn, acc_trn, vals_trn = svm_predict(Ytr, Xtr, model)
        Ein[digit] = 100.0 - acc_trn[0]

        Xtest, Ytest = read_one_versus_all('features.test.txt', digit)
        labels_test, acc_test, vals_test = svm_predict(Ytest, Xtest, model)
        Eout[digit] = 100.0 - acc_test[0]
    # disp_min_and_max_error(Ein, "Ein")
    disp_min_and_max_error(Eout, "Eout")


def one_versus_another(first_digit=1, second_digit=5, degree=2):
    """ First digit versus another (second) digit implementation without the cross-validation.
        kernel: polynomial: (gamma * (u' .dot v) + coef) ^ degree
    """
    print "\n*****degree: %i****" % degree
    insample_errors = []
    outofsample_errors = []
    support_vectors = []
    Xtr, Ytr = read_one_versus_another('features.train.txt', first_digit, second_digit)
    Xtest, Ytest = read_one_versus_another('features.test.txt', first_digit, second_digit)
    # c_list = [0.001, 0.01, 0.1, 1.0]
    c_list = [0.0001, 0.001, 0.01, 1.0]
    for C in c_list:
        const_info = "-c " + str(C) + " -t 1 -d " + str(degree) + " -g 1 -r 1 -h 0 -q"
        param = svm_parameter(const_info)
        problem = svm_problem(Ytr, Xtr)
        model = svm_train(problem, param)
        support_vectors.append(len(model.get_SV()))

        acc_trn = svm_predict(Ytr, Xtr, model)[1][0]
        insample_errors.append(100.0 - acc_trn)

        acc_test = svm_predict(Ytest, Xtest, model)[1][0]
        outofsample_errors.append(100.0 - acc_test)

    print "Ein: ", insample_errors
    print "Eout: ", outofsample_errors
    print "#SV: ", support_vectors
    plt.figure(degree)
    c_points = [0.1, 0.5, 1, 2]
    plt.subplot(211)
    plt.title("degree: %i" % degree)
    plt.ylabel('error, %')
    plt.plot(c_points, insample_errors, 'b-')
    plt.plot(c_points, outofsample_errors, 'g-')
    plt.legend(['in-sample error', 'out-of-sample error'])
    plt.plot(c_points, insample_errors, 'bo')
    plt.plot(c_points, outofsample_errors, 'go')

    plt.subplot(212)
    plt.ylabel('#SV')
    plt.xlabel('constraints C')
    plt.plot(c_points, support_vectors, 'y-')
    plt.plot(c_points, support_vectors, 'yo')
    plt.show()


def run_cross_validation(first_digit=1, second_digit=5):
    """ Picks the best constraint for C among constraints_list,
        which yields the lowest cross-validation error Eval.
    """
    Xtr, Ytr = read_one_versus_another('features.train.txt', first_digit, second_digit)
    constraints_list = [0.0001, 0.001, 0.01, 0.1, 1.0]
    Eval_aver = np.array([.0, .0, .0, .0, .0])
    constraints_id_list = [0, 0, 0, 0, 0]
    Niters = 100
    for dummy_i in range(Niters):
        crossvalid_errors = np.array([])
        for idC in range(len(constraints_list)):
            const_info = "-c " + str(constraints_list[idC]) + " -t 1 -d 2 -g 1 -r 1 -h 0 -v 10 -q"
            param = svm_parameter(const_info)
            problem = svm_problem(Ytr, Xtr)
            Eval = svm_train(problem, param)
            crossvalid_errors = np.append(crossvalid_errors, 100.0 - Eval)
        Eval_aver += crossvalid_errors / Niters
        Eval_min_id = crossvalid_errors.argmin()
        constraints_id_list[Eval_min_id] += 1
    print "Eval_aver (in %%): ", Eval_aver
    print "constraints_id_list: ", constraints_id_list


def run_rbf_kernel(first_digit=1, second_digit=5):
    """ Evaluates the in-sample and out-of-sample performance without
            the regularization for the one-versus-another SVM classifier.
		kernel: RBF: exp(- gamma * ||u - v||^2)
        Prints out the best constraints C_in and C_out that yield
            the lowest errors Ein and Eout respectively.
    """
    insample_errors = []
    outofsample_errors = []
    Xtr, Ytr = read_one_versus_another('features.train.txt', first_digit, second_digit)
    Xtest, Ytest = read_one_versus_another('features.test.txt', first_digit, second_digit)
    c_list = [0.01, 1.0, 1e2, 1e4, 1e6]
    for C in c_list:
        const_info = "-c " + str(C) + " -t 2 -g 1 -h 0 -q"
        param = svm_parameter(const_info)
        problem = svm_problem(Ytr, Xtr)
        model = svm_train(problem, param)

        acc_trn = svm_predict(Ytr, Xtr, model)[1][0]
        insample_errors.append(100.0 - acc_trn)

        acc_test = svm_predict(Ytest, Xtest, model)[1][0]
        outofsample_errors.append(100.0 - acc_test)

    train_id_min = np.array(insample_errors).argmin()
    test_id_min = np.array(outofsample_errors).argmin()
    print "Ein[%i] = %g%%" % (train_id_min, insample_errors[train_id_min])
    print "Eout[%i] = %g%%" % (test_id_min, outofsample_errors[test_id_min])


one_versus_all()