import numpy as np
from numpy import linalg as LA

epsilon = 10 ** -8
N = 1000
Max_LS_iter = 20
mu = 10 ** -4
v = 0.9
y = 10 ** -4

parameters = [epsilon, N, Max_LS_iter, mu, v, y]


def backtracking(mu, p_k, x_k):
    n = 0
    alpha = 1
    while function(x_k + (alpha * p_k)) > function(x_k) + (mu * alpha) * np.matmul(np.transpose(gradient), p_k):
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


def modified_newton(epsilon, N, Max_LS_iter, mu, v, y, x0):
    k = 0
    x_k = x0
    h = hessian(x_k)

    while LA.norm(gradient(x_k))/ (1 + abs(function(x_k)) > epsilon) and k <= N:
        #Check that hessian is positive definite
        if LA.eig(h) <= 0:
            PD_adjust = np.amin(LA.eig(h)) + 0.01
            h = h + PD_adjust * np.identity(2, dtype=float) #acceptable dimensions
        p_k = LA.solve(h,(-1 * gradient(x_k)))
        alpha = backtracking()
        x_k = x_k + alpha * p_k
        k += 1
        h = hessian(x_k)
    return x_k, function(x_k), k

def quasi():
    k = 0
    x_k = x0
    b_k = np.identity() #acceptable dimensions

    for i in range(N):
        p_k = LA.solve(b_k, (-1 * gradient(x_k)))
        alpha = backtracking(epsilon, N, Max_LS_iter, mu, v, y, p_k, x_k)
        x_k1 = x_k + (alpha * p_k)
        s_k = x_k1 - x_k
        y_k = gradient(x_k1) - gradient(x_k)
        b_k = b_k - (np.matmul(b_k, s_k) * np.matmul(b_k, s_k).transpose())/ (s_k.transpose() * np.matmul(b_k, s_k))
        b_k = b_k + (np.matmul(y_k, y_k.transpose()))/np.matmul(y_k.transpose(), s_k)
    return x_k, function(x_k), k
