import numpy as np
from numpy import linalg as LA
epsilon = 10 ** -8
N = 1000
Max_LS_iter = 20
mu = 10 ** -4
v = 0.9
y = 10 ** -4


def function(num1, num2, c):
    return ((num1 - 1) ** 2) + ((num2 -1) ** 2) + (c *((num1 ** 2) + (num2 ** 2) - 0.25))
def gradient(num1, num2, c):
    a = ((4 * (num1 ** 3) * c) + (4*c * num1 * (num2 ** 2)) + (2 * num1) - (num1 * c)) - 2
    b = ((4 * (num2 ** 3) * c) + (4*c * (num1 ** 2) * num2) + (2 * num2) - (num2 * c)) - 2
    return np.array([a,b])

def hessian(num1, num2, c):
    a = (12 * (num1 ** 2) * c) + (4 * c * (num2 ** 2)) + 2 - c
    b = 8 * c * num1 * num2
    c = 8 * c * num1 * num2
    d = (12 * (num2 ** 2) * c) + (4 * c * (num1 ** 2)) + 2 - c
    return np.array([[a,b],[c,d]])




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

def gradient_descent(x0, epsilon=10 ** -8, N=1000, Max_LS_iter=20, mu=10 ** -4, v=0.9, y=10 ** -4):
    k = 0
    x_k = x0
    while (LA.norm(gradient(x_k[0], x_k[1], c),2) / (1 + abs(function(x_k[0], x_k[1], c)))) > epsilon and k <= N:  #figure out gradient information on how to plug stuff into it
        p_k = -gradient(x_k[0], x_k[1], c)
        alpha = backtracking(mu, p_k, x_k)
        x_k = x_k + (alpha * p_k)
        k += 1
    if k > N:
        print("Number of iterations exceeded limit: ", N)
        return x_k, function(x_k[0], x_k[1], c), k
    else:
        return x_k, function(x_k[0], x_k[1], c), k

x0 = np.array([1,-1])
c = int(input("Enter C Value: "))
print(gradient_descent(x0))

# c = 1 Number of iterations exceeded limit:  1000 (array([ 0.9966822 , -0.99477701]), 5.712103061108813, 1001)
# c = 10 Number of iterations exceeded limit:  1000 (array([ 0.9683653, -0.9664869]), 20.08635417559048, 1001)
# c = 100 Number of iterations exceeded limit:  1000 (array([ 0.78172209, -0.78000998]), 100.16658142980685, 1001)