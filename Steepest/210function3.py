import numpy as np
from numpy import linalg as LA
epsilon = 10 ** -8
N = 1000
Max_LS_iter = 20
mu = 10 ** -4
v = 0.9
y = 10 ** -4


def function(num1,num2):
    return (100 * (num2 - (num1 ** 2)**2) + (1 - num1) ** 2)
def gradient(num1,num2):
    a = (400 * num1 ** 3) - (400 * num1 * num2) + (2 * num1) - 2
    b = (200 * num2) - (200 * num1 ** 2)
    vector = [a,b]
    vector = np.asarray(vector).transpose()
    return vector
def hessian(num1,num2):
    a = (3 * num1 ** 2) - (num2) + 2
    b = -num1
    c = -num1
    d = 0.5
    return np.array([[a,b],[c,d]])


def backtracking(mu, p_k, x_k, Max_LS_iter=20):
    n = 0
    alpha = 1
    arg1 = np.add(x_k, (alpha * p_k))
    arg2 = x_k
    while function(arg1[0], arg1[1]) > (function(arg2[0], arg2[1]) + (mu * alpha) * np.matmul(np.transpose(gradient(arg2[0], arg2[1])), p_k)) and n <= Max_LS_iter:
        alpha = alpha / 2
        n += 1
        print(n)
    return alpha

def gradient_descent(x0, epsilon=10 ** -8, N=1000, Max_LS_iter=20, mu=10 ** -4, v=0.9, y=10 ** -4):
    data = [['iteration', 'norm of gradient', 'step size', 'x_k']]
    k = 0
    x_k = x0
    while (LA.norm(gradient(x_k[0], x_k[1]),2) / (1 + abs(function(x_k[0], x_k[1])))) > epsilon and k <= N:  #figure out gradient information on how to plug stuff into it
        p_k = -gradient(x_k[0], x_k[1])
        alpha = backtracking(mu, p_k, x_k)
        x_k = x_k + (alpha * p_k)
        k += 1
        data.append([k, LA.norm(gradient(x_k[0], x_k[1])), alpha, x_k])
    if k > N:
        print("Number of iterations exceeded limit: ", N)
        return data
    else:
        return data

x0 = np.array([-1.2,1])
data = gradient_descent(x0)
if len(data) > 15:
    a = data[0:11]
    a.extend(data[len(data) - 6:len(data) - 1])
    for item in a:
        print(item)
else:
    for item in data:
        print(item)
#output
# ['iteration', 'norm of gradient', 'step size', 'x_k']
# [1, 3934539279.628693, 1, array([214.4,  89. ])]
# [2, 2.436340773593038e+31, 1, array([-3.93452837e+09,  9.17576100e+06])]
