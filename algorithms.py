import numpy as np
from numpy import linalg as LA

epsilon = 10 ** -8
N = 1000
Max_LS_iter = 20
mu = 10 ** -4
v = 0.9
y = 10 ** -4

parameters = [epsilon, N, Max_LS_iter, mu, v, y]


def backtracking(mu, p_k, x_k, Max_LS_iter=20):
    n = 0
    alpha = 1
    arg1 = np.add(x_k, (alpha * p_k))
    arg2 = x_k
    while function(arg1[0], arg1[1],c) > (function(arg2[0], arg2[1], c) + (mu * alpha) * np.matmul(np.transpose(gradient(arg2[0], arg2[1], c)), p_k)) and n <= Max_LS_iter:
        alpha = alpha / 2
        n += 1
        print(n)
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


def modified_newton(x0, epsilon=10 ** -8, N=1000, Max_LS_iter=20, mu=10 ** -4, v=0.9, y=10 ** -4):
    k = 0
    x_k = x0
    h = hessian(x_k[0],x_k[1],c)
    print("h = ",h)
    print((LA.norm(gradient(x_k[0],x_k[1], c), 2)))
    print((1 + abs(function(x_k[0],x_k[1], c))))
    while (LA.norm(gradient(x_k[0],x_k[1], c),2) / (1 + abs(function(x_k[0],x_k[1], c)))) > epsilon and k <= N:
        #Check that hessian is positive definite
        eig_values, eig_vectors = LA.eig(h)
        shape = eig_values.shape
        print("eig values =",eig_values)
        print(np.less_equal(eig_values,0).any())
        if np.less_equal(eig_values,0).any() =='True':
            PD_adjust = abs(np.amin(eig_values)) + 0.01
            print("PD_adjust =", PD_adjust)
            print((PD_adjust * np.identity(2, dtype=float)))
            h = np.add(h, (PD_adjust * np.identity(2, dtype=float))) #acceptable dimensions
            print("h changed = ", h)
        p_k = LA.solve(h,(-1 * gradient(x_k[0],x_k[1],c)))
        print("p_k = ", p_k)
        alpha = backtracking(mu, p_k, x_k)
        x_k = x_k + alpha * p_k
        k += 1
        h = hessian(x_k[0] , x_k[1],c)
        print("hi")
    return x_k, function(x_k[0],x_k[1],c), k

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
