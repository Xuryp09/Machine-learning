import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def GPR(point_x, point_y, x_star, var, lengthscale, alpha, beta):
    C = RQkernel(point_x, point_x, var, lengthscale, alpha) + np.eye(len(point_x)) / beta
    k = RQkernel(point_x, x_star, var, lengthscale, alpha) 
    k_star = RQkernel(x_star, x_star, var, lengthscale, alpha) + np.eye(len(x_star)) / beta

    mu_star = np.transpose(k) @ np.linalg.inv(C) @ point_y
    var_star = k_star - np.transpose(k) @ np.linalg.inv(C) @ k

    return mu_star, var_star

def RQkernel(x_a, x_b, var, lengthscale, alpha):
    return var * np.power(1 + cdist(x_a, x_b, 'sqeuclidean') / (2 * alpha * np.power(lengthscale, 2)), -alpha)

def negative_marginal_log_likelihood(theta, point_x, point_y, beta):
    C = RQkernel(point_x, point_x, theta[0], theta[1], theta[2]) + np.eye(len(point_x)) / beta
    result = 0.5 * (np.log(np.linalg.det(C)) + np.transpose(point_y) @ np.linalg.inv(C) @ point_y + len(point_x) * np.log(2 * np.pi))
    return result[0]

def visualization(point_x, point_y, x_star, mu_star, var_star, var, lengthscale, alpha):
    plt.title("Gaussian Process with var: {0:.2f}, scale length: {1:.2f}, alpha: {2:.2f}".format(var, alpha, lengthscale))
    plt.xlim(-60, 60)
    double_std = 1.96 * np.sqrt(np.diag(var_star))
    plt.fill_between(x_star.ravel(), mu_star.ravel() + double_std, mu_star.ravel() - double_std, color = 'pink', alpha = 0.3)
    plt.scatter(point_x, point_y, color = 'b', s = 20)
    plt.plot(x_star.ravel(), mu_star.ravel(), 'r')
    plt.show()

if __name__ == '__main__':

    # load input data
    file_path = './data/input.data' 
    point_x = []
    point_y = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.split()
            point_x.append(float(values[0]))        
            point_y.append(float(values[1]))
    point_x = np.array(point_x).reshape(-1, 1)
    point_y = np.array(point_y).reshape(-1, 1)
    beta = 5

    # setting for kernel 
    var = 1.0
    lengthscale = 1.0
    alpha = 1.0
  
    # Gaussian process regression
    x_star = np.linspace(-60.0, 60.0, 1000).reshape(-1, 1)
    mu_star, var_star = GPR(point_x, point_y, x_star, var, lengthscale, alpha, beta)
    visualization(point_x, point_y, x_star, mu_star, var_star, var, lengthscale, alpha)
    
    # Optimize the kernel parameters
    opt_param = minimize(negative_marginal_log_likelihood, [var, lengthscale, alpha], args = (point_x, point_y, beta))
    mu_opt, var_opt = GPR(point_x, point_y, x_star, opt_param.x[0], opt_param.x[1], opt_param.x[2], beta)
    visualization(point_x, point_y, x_star, mu_opt, var_opt, opt_param.x[0], opt_param.x[1], opt_param.x[2])
    