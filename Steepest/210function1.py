import numpy as np
from numpy import linalg as LA


epsilon = 10 ** -8
N = 1000
Max_LS_iter = 20
mu = 10 ** -4
v = 0.9
y = 10 ** -4


def function(num1, num2, num3):
    return ((num1 ** 2) + (num2 ** 2) + (num3 ** 2))
def gradient(num1, num2, num3):
    vector = [2 * num1, 2 * num2, 2 * num3]
    vector = np.asarray(vector).transpose()
    return vector
def hessian(num1, num2, num3):
    return 2 * np.identity(3, dtype=float)

def backtracking(mu, p_k, x_k, Max_LS_iter=20):
    n = 0
    alpha = 1
    arg1 = np.add(x_k, (alpha * p_k))
    arg2 = x_k
    while function(arg1[0], arg1[1], arg1[2]) > (function(arg2[0], arg2[1], arg2[2]) + (mu * alpha) * np.matmul(np.transpose(gradient(arg2[0], arg2[1], arg2[2])), p_k)) and n <= Max_LS_iter:
        alpha = alpha / 2
        n += 1
    return alpha

def gradient_descent(function_value_list, epsilon, N, Max_LS_iter, mu, v, y, gradient_value_list, x0):
    k = 0
    x_k = x0
    while LA.norm(gradient(x_k)) / (1 + abs(function(x_k)) > epsilon) and k <= N:  #figure out gradient information on how to plug stuff into it
        p_k = -gradient(x_k)
        alpha = backtracking(mu, v, y, p_k, x_k)
        x_k = x_k + (alpha * p_k)
        k += 1
    if k > N:
        print("Number of iterations exceeded limit *N", N)
        return x_k, function(x_k), k
    else:
        return x_k, function(x_k), k