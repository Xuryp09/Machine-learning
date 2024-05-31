import numpy as np
import matplotlib.pyplot as plt
from libsvm.svmutil import *
from scipy.spatial.distance import cdist

def load_x(filename):
    file = open(filename, 'r')
    lines = file.read()
    x = [[float(pixel) for pixel in line.split(',') if pixel.strip()] for line in lines.split('\n') if line.strip()]
    return x

def load_y(filename):
    with open(filename, 'r') as file:
        lines = file.read()
        y = [int(line) for line in lines.split('\n') if line.strip()]
    return y
    
def compare_diff_kernel(x_train, x_test, y_train, y_test):
    
    print("Linear kernel:")
    linear_model = svm_train(y_train, x_train, '-t 0 -q')
    svm_predict(y_test, x_test, linear_model)
    
    print("Polynomial kernel:") 
    poly_model = svm_train(y_train, x_train, '-t 1 -q')
    svm_predict(y_test, x_test, poly_model)

    print("Rbf kernel:")
    rbf_model = svm_train(y_train, x_train, '-t 2 -q')
    svm_predict(y_test, x_test, rbf_model)

def grid_search(x_train, y_train):
    C = [0.001, 0.01, 0.1, 1, 10]
    G = [0.001, 0.01, 0.1, 1, 10]

    for kernel in range(0 ,3):
        if (kernel == 0):
            opt_param_linear_pred = ""
            opt_param_linear = ""
            opt_accuracy_linear = 0
            count = 1
            for cost in C:
                param_linear = f"-t 0 -c {cost} -v 3 -q"
                param_linear_pred = f"-t 0 -c {cost} -q"
                print("\n Linear kernel ", count)
                print(param_linear)
                count += 1
                accuracy_linear = svm_train(y_train, x_train, param_linear)
                if (accuracy_linear > opt_accuracy_linear):
                    opt_param_linear_pred = param_linear_pred
                    opt_accuracy_linear = accuracy_linear
                    opt_param_linear = param_linear
        
        elif (kernel == 1):
            opt_param_poly_pred = ""
            opt_param_poly = ""
            opt_accuracy_poly = 0
            count = 1 
            for cost in C:
                for gamma in G:
                    for degree in range(2, 4):
                        for constant in range(0, 2):
                            param_poly = f"-t 1 -c {cost} -g {gamma} -d {degree} -r {constant} -v 3 -q"
                            param_poly_pred = f"-t 1 -c {cost} -g {gamma} -d {degree} -r {constant} -q"
                            print("\n Polynomial kernel ", count)
                            print(param_poly)
                            count += 1
                            accuracy_poly = svm_train(y_train, x_train, param_poly)
                            if (accuracy_poly > opt_accuracy_poly):
                                opt_param_poly_pred = param_poly_pred
                                opt_accuracy_poly = accuracy_poly
                                opt_param_poly = param_poly
        
        else: 
            opt_param_rbf_pred = ""
            opt_param_rbf = ""
            opt_accuracy_rbf = 0
            count = 1
            for cost in C:
                for gamma in G:
                    param_rbf = f"-t 2 -c {cost} -g {gamma} -v 3 -q"
                    param_rbf_pred = f"-t 2 -c {cost} -g {gamma} -q"
                    print("\n Rbf kernel ", count)
                    print(param_rbf)
                    count += 1
                    accuracy_rbf = svm_train(y_train,  x_train, param_rbf)
                    if (accuracy_rbf > opt_accuracy_rbf):
                        opt_param_rbf_pred = param_rbf_pred
                        opt_accuracy_rbf = accuracy_rbf
                        opt_param_rbf = param_rbf
    
    print("Linear kernel:")
    print(f'Optimal cross validation accuracy: {opt_accuracy_linear}')
    print(f'Optimal parameter: {opt_param_linear}')
    
    print("Polynomial kernel:")
    print(f'Optimal cross validation accuracy: {opt_accuracy_poly}')
    print(f'Optimal parameter: {opt_param_poly}')

    print("Rbf kernel:")
    print(f'Optimal cross validation accuracy: {opt_accuracy_rbf}')
    print(f'Optimal parameter: {opt_param_rbf}')

    print("")
    print("Linear kernel:")
    linear_model = svm_train(y_train, x_train, opt_param_linear_pred)
    svm_predict(y_test, x_test, linear_model)
    
    print("Polynomial kernel:") 
    poly_model = svm_train(y_train, x_train, opt_param_poly_pred)
    svm_predict(y_test, x_test, poly_model)

    print("Rbf kernel:")
    rbf_model = svm_train(y_train, x_train, opt_param_rbf_pred)
    svm_predict(y_test, x_test, rbf_model)

def combine_linear_rbf(x_train, x_test, y_train, y_test):
    C = [0.001, 0.01, 0.1, 1, 10]
    G = [0.001, 0.01, 0.1, 1, 10]

    opt_param_combine_pred = ""
    opt_param_combine = ""
    opt_accuracy_combine = 0
    opt_gamma = 1
    count = 1

    linear_kernel = x_train @ np.transpose(x_train)
    for cost in C:
        for gamma in G:
            rbf_kernel = np.exp(- gamma * cdist(x_train, x_train, 'sqeuclidean'))
            combine_kernel = linear_kernel + rbf_kernel
            combine_kernel = np.hstack((np.arange(1, 5001).reshape(-1, 1), combine_kernel))

            param_combine = f"-t 4 -c {cost} -v 3 -q"
            param_combine_pred = f"-t 4 -c {cost} -q"
            print("\nLinear + rbf kernel ", count)
            print(param_combine, f" -g {gamma}")
            count += 1
            accuracy_combine = svm_train(y_train, combine_kernel, param_combine)
            if (accuracy_combine > opt_accuracy_combine):
                opt_param_combine_pred = param_combine_pred
                opt_accuracy_combine = accuracy_combine
                opt_param_combine = param_combine
                opt_gamma = gamma
            
    print("Linear + rbf kernel:")
    print(f'Optimal cross validation accuracy: {opt_accuracy_combine}')
    print(f'Optimal parameter: {opt_param_combine}')
    print(f'Optimal gamma: {opt_gamma}') 

    rbf_kernel = np.exp(- opt_gamma * cdist(x_train, x_train, 'sqeuclidean'))
    combine_kernel = linear_kernel + rbf_kernel
    combine_kernel = np.hstack((np.arange(1, 5001).reshape(-1, 1), combine_kernel))
    combine_model = svm_train(y_train, combine_kernel, opt_param_combine_pred)

    test_linear_kernel = x_test @ np.transpose(x_train)
    test_rbf_kernel = np.exp(- opt_gamma * cdist(x_test, x_train, 'sqeuclidean'))
    test_combine_kernel = test_linear_kernel + test_rbf_kernel
    test_combine_kernel = np.hstack((np.arange(1, 2501).reshape(-1, 1), test_combine_kernel))
    svm_predict(y_test, test_combine_kernel, combine_model)

if __name__ == '__main__':
    
    # Load data
    x_train = load_x('./data/X_train.csv')
    x_test = load_x('./data/X_test.csv')
    y_train = load_y('./data/Y_train.csv')
    y_test = load_y('./data/Y_test.csv')

    # Task 1 : Compare to each kind of kernels
    compare_diff_kernel(x_train, x_test, y_train, y_test)

    # Task 2 : Do the grid search for finding parameters of best performing model
    grid_search(x_train, y_train)

    # Task 3 : Use linear kernel + RBF kernel together
    combine_linear_rbf(x_train, x_test, y_train, y_test)