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

# def backtracking(mu, p_k, x_k):
#     n = 0
#     alpha = 1
#     while function(x_k + (alpha * p_k)) > function(x_k) + (mu * alpha) * np.matmul(np.transpose(gradient), p_k):
#         alpha = alpha / 2
#         n += 1
#     return alpha

def modified_newton(x0, epsilon=10 ** -8, N=1000, Max_LS_iter=20, mu=10 ** -4, v=0.9, y=10 ** -4):
    k = 0
    x_k = x0
    h = hessian(x_k[0],x_k[1],x_k[2])

    while LA.norm(gradient(x_k[0],x_k[1],x_k[2]))/ (1 + abs(function(x_k[0],x_k[1],x_k[2])) > epsilon) and k <= N:
        #Check that hessian is positive definite
        if LA.eig(h) <= 0:
            PD_adjust = np.amin(LA.eig(h)) + 0.01
            h = h + PD_adjust * np.identity(3, dtype=float) #acceptable dimensions
        p_k = LA.solve(h,(-1 * gradient(x_k[0],x_k[1],x_k[2])))
        alpha = backtracking(mu, p_k, x_k)
        x_k = x_k + alpha * p_k
        k += 1
        h = hessian(x_k)
    return x_k, function(x_k), k

x0 = np.array([1,1,1])
print(x0)
print(x0[0])