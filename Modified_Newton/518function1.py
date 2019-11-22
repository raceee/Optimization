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
        n += 1
    return alpha

def modified_newton(x0, epsilon=10 ** -8, N=1000, Max_LS_iter=20, mu=10 ** -4, v=0.9, y=10 ** -4):
    data = [['iteration', 'function at x_k', 'norm of gradient', 'step size', 'x_k']]
    k = 0
    x_k = x0
    h = hessian(x_k[0],x_k[1],x_k[2])
    print("h = ",h)
    print((LA.norm(gradient(x_k[0],x_k[1],x_k[2]), 2)))
    print((1 + abs(function(x_k[0],x_k[1],x_k[2]))))
    while (LA.norm(gradient(x_k[0],x_k[1],x_k[2]),2) / (1 + abs(function(x_k[0],x_k[1],x_k[2])))) > epsilon and k <= N:
        #Check that hessian is positive definite
        eig_values, eig_vectors = LA.eig(h)
        shape = eig_values.shape
        print("eig =", eig_values)
        print(np.less_equal(eig_values,0).any())
        if np.less_equal(eig_values,0).any() =='True':
            PD_adjust = abs(np.amin(eig_values)) + 0.01
            print("PD_adjust =", PD_adjust)
            h = np.add(h, (PD_adjust * np.identity(3, dtype=float))) #acceptable dimensions
            print("h = ", h)
        p_k = LA.solve(h,(-1 * gradient(x_k[0],x_k[1],x_k[2])))
        print("p_k = ", p_k)
        alpha = backtracking(mu, p_k, x_k)
        x_k = x_k + alpha * p_k
        k += 1
        h = hessian(x_k[0], x_k[1], x_k[2])
        data.append([k, function(x_k[0],x_k[1],x_k[2]), LA.norm(gradient(x_k[0],x_k[1], x_k[2])), alpha, x_k])
    return data

x0 = np.array([1,1,1])
data = modified_newton(x0)
if len(data) > 15:
    a = data[0:11]
    a.extend(data[len(data) - 6:len(data) - 1])
    for item in a:
        print(a)
else:
    for item in data:
        print(item)



# output
# ['iteration', 'function at x_k', 'norm of gradient', 'step size', 'x_k']
# [1, 0.0, 0.0, 1, array([0., 0., 0.])]