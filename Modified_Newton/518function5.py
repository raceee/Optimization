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

x0 = np.array([1,-1])
c = int(input("Enter C Value: "))
print(modified_newton(x0))

# c = 1 (array([ 0.55833181, -1.06334484]), 5.644899351245723, 1001)
# c = 10 (array([ 0.16172262, -1.09017924]), 14.718008106821928, 1001)
# c = 100 (array([ 0.14194704, -1.07904102]), 98.50651403324103, 1001)