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

def modified_newton(x0, epsilon=10 ** -8, N=1000, Max_LS_iter=20, mu=10 ** -4, v=0.9, y=10 ** -4):
    k = 0
    x_k = x0
    h = hessian(x_k[0],x_k[1])
    print("h = ",h)
    print((LA.norm(gradient(x_k[0],x_k[1]), 2)))
    print((1 + abs(function(x_k[0],x_k[1]))))
    while (LA.norm(gradient(x_k[0],x_k[1]),2) / (1 + abs(function(x_k[0],x_k[1])))) > epsilon and k <= N:
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
        p_k = LA.solve(h,(-1 * gradient(x_k[0],x_k[1])))
        print("p_k = ", p_k)
        alpha = backtracking(mu, p_k, x_k)
        x_k = x_k + alpha * p_k
        k += 1
        h = hessian(x_k[0] , x_k[1])
        print("hi")
    return x_k, function(x_k[0],x_k[1]), k

x0 = np.array([1.9,-2])

print(modified_newton(x0))

#x0 makes the hessian singular and thus it breaks this function.
#trying (1.9,-2) outputs (array([ 1.88876792, -1.98857718]), 3.9545384253128253, 1001)