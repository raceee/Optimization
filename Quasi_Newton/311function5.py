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

def quasi(x0, epsilon = 10 ** -8, N = 1000, Max_LS_iter = 20, mu = 10 ** -4, v = 0.9, y = 10 ** -4):
    k = 0
    x_k = x0
    b_k = np.identity(2, dtype="float") #acceptable dimensions

    while k <= N:
        p_k = LA.solve(b_k, (-1 * gradient(x_k[0], x_k[1],c)))
        print("p_k = ", p_k)
        alpha = backtracking(mu, p_k, x_k, Max_LS_iter=20)
        mag_and_direc = alpha * p_k
        x_k1 = np.add(x_k, mag_and_direc)
        print("x_k1 = ", x_k1)
        s_k = np.subtract(x_k1, x_k)
        print("s_k = ", s_k)
        y_k = np.subtract(gradient(x_k1[0], x_k1[1], c), gradient(x_k[0], x_k[1], c))
        print("y_k = ", y_k)
        numerator = np.matmul(np.matmul(b_k, s_k), np.matmul(b_k, s_k).transpose())
        denominator = np.matmul(s_k.transpose(), np.matmul(b_k, s_k))
        a = numerator / denominator
        b = np.matmul(y_k, y_k.transpose()) / np.matmul(y_k.transpose(), s_k)
        print("b =", b)
        ab = a + b
        b_k = np.subtract(b_k, (ab * np.identity(2, dtype="float")))
        x_k = x_k1
        k += 1
        print(k)
    return x_k, function(x_k[0], x_k[1], c), k

x0 = np.array([1,-1])
c = int(input("Enter C Value: "))

x_k_out, fx_k, k = quasi(x0)

print("x_k = {}, function at x_k = {}, iterations = {}".format(x_k_out,fx_k, k))
#output
# c = 1, x_k = [ 1.0001317  -1.00020694], function at x_k = 5.7515051816684695, iterations = 1001
# c = 10, x_k = [ 1.00011053 -1.00011684], function at x_k = 21.505014961160693, iterations = 1001
# c = 100, x_k = [ 0.99981117 -0.99981009], function at x_k = 178.92349862506282, iterations = 1001