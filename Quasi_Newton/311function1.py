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
        print("alpha = ", alpha)
        n += 1
    return alpha

def quasi(x0, epsilon = 10 ** -8, N = 1000, Max_LS_iter = 20, mu = 10 ** -4, v = 0.9, y = 10 ** -4):
    k = 0
    x_k = x0
    b_k = np.identity(3, dtype="float") #acceptable dimensions

    for i in range(N):
        p_k = LA.solve(b_k, (-1 * gradient(x_k[0], x_k[1], x_k[2])))
        alpha = backtracking(mu, p_k, x_k, Max_LS_iter=20)
        mag_and_direc = alpha * p_k
        x_k1 = np.add(x_k, mag_and_direc)
        s_k = np.subtract(x_k1, x_k)
        y_k = np.subtract(gradient(x_k1[0], x_k1[1], x_k1[2]), gradient(x_k[0], x_k[1], x_k[2]))
        numerator = np.matmul(np.matmul(b_k, s_k), np.matmul(b_k, s_k).transpose())
        denominator = np.matmul(s_k.transpose(), np.matmul(b_k, s_k))
        a = numerator / denominator
        b = np.matmul(y_k, y_k.transpose()) / np.matmul(y_k.transpose(), s_k)
        ab = a + b
        b_k = np.subtract(b_k, (ab * np.identity(3, dtype="float")))
        x_k = x_k1
        k = i
    return x_k, function(x_k[0], x_k[1], x_k[2]), k

x0 = np.array([1,1,1])

x_k_out, fx_k, k = quasi(x0)

print("x_k = {}, function at x_k = {}, iterations = {}".format(x_k_out,fx_k,k))