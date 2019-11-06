import numpy as np
from numpy import linalg as LA
epsilon = 10 ** -8
N = 1000
Max_LS_iter = 20
mu = 10 ** -4
v = 0.9
y = 10 ** -4


def function(num1, num2):
    return (((num1 + num2) ** 4) + (num2 ** 2))
def gradient(num1, num2):
    a = (4 * (num1 + num2) ** 3)
    b = (4 * (num1 + num2) **3) + 2 * num2
    return np.array([a,b])
def hessian(num1, num2):
    a = (num1 + num2) **2
    b = (num1 + num2) **2
    c = (num1 + num2) **2
    d = (num1 + num2) **2 + (1/6)
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
    k = 0
    x_k = x0
    while (LA.norm(gradient(x_k[0], x_k[1]),2) / (1 + abs(function(x_k[0], x_k[1])))) > epsilon and k <= N:  #figure out gradient information on how to plug stuff into it
        p_k = -gradient(x_k[0], x_k[1])
        alpha = backtracking(mu, p_k, x_k)
        x_k = x_k + (alpha * p_k)
        k += 1
    if k > N:
        print("Number of iterations exceeded limit: ", N)
        return x_k, function(x_k[0], x_k[1]), k
    else:
        return x_k, function(x_k[0], x_k[1]), k

x0 = np.array([2,-2])
print(gradient_descent(x0))

# Number of iterations exceeded limit:  1000
# (array([ 2.        , -1.99809165]), 3.992370258345243, 1001)