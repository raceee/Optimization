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
    data = [['iteration', 'norm of gradient', 'step size', 'x_k']]
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
        data.append([k, LA.norm(gradient(x_k[0], x_k[1])), alpha, x_k])
    return data

x0 = np.array([1.9,-2])
data = modified_newton(x0)
if len(data) > 15:
    a = data[0:11]
    a.extend(data[len(data) - 6:len(data) - 1])
    for item in a:
        print(item)
else:
    for item in data:
        print(item)

# output
# ['iteration', 'norm of gradient', 'step size', 'x_k']
# [1, 4.003979086918333, 4.76837158203125e-07, array([ 1.89998875, -1.99998856])]
# [2, 4.003956175966265, 4.76837158203125e-07, array([ 1.89997749, -1.99997711])]
# [3, 4.003933265145294, 4.76837158203125e-07, array([ 1.89996624, -1.99996567])]
# [4, 4.003910354455421, 4.76837158203125e-07, array([ 1.89995499, -1.99995422])]
# [5, 4.003887443896643, 4.76837158203125e-07, array([ 1.89994373, -1.99994278])]
# [6, 4.003864533468961, 4.76837158203125e-07, array([ 1.89993248, -1.99993134])]
# [7, 4.003841623172373, 4.76837158203125e-07, array([ 1.89992123, -1.99991989])]
# [8, 4.003818713006879, 4.76837158203125e-07, array([ 1.89990997, -1.99990845])]
# [9, 4.003795802972478, 4.76837158203125e-07, array([ 1.89989872, -1.99989701])]
# [10, 4.003772893069169, 4.76837158203125e-07, array([ 1.89988747, -1.99988556])]
# [996, 3.9812473966870767, 4.76837158203125e-07, array([ 1.88882386, -1.98863407])]
# [997, 3.981224615806783, 4.76837158203125e-07, array([ 1.88881267, -1.98862269])]
# [998, 3.981201835056843, 4.76837158203125e-07, array([ 1.88880149, -1.98861131])]
# [999, 3.981179054437255, 4.76837158203125e-07, array([ 1.8887903 , -1.98859993])]
# [1000, 3.9811562739480197, 4.76837158203125e-07, array([ 1.88877911, -1.98858856])]