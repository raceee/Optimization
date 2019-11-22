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
    data = [['iteration', 'function at x_k','norm of gradient', 'step size', 'x_k']]
    k = 0
    x_k = x0
    while (LA.norm(gradient(x_k[0], x_k[1]),2) / (1 + abs(function(x_k[0], x_k[1])))) > epsilon and k <= N:  #figure out gradient information on how to plug stuff into it
        p_k = -gradient(x_k[0], x_k[1])
        alpha = backtracking(mu, p_k, x_k)
        x_k = x_k + (alpha * p_k)
        k += 1
        data.append([k, function(x_k[0],x_k[1]), LA.norm(gradient(x_k[0], x_k[1])), alpha, x_k])
    if k > N:
        print("Number of iterations exceeded limit: ", N)
        return data
    else:
        return data

x0 = np.array([2,-2])
data = gradient_descent(x0)
if len(data) > 15:
    a = data[0:11]
    a.extend(data[len(data) - 6:len(data) - 1])
    for item in a:
        print(item)
else:
    for item in data:
        print(item)

# ['iteration', 'function at x_k', 'norm of gradient', 'step size', 'x_k']
# [1, 3.9999923706091067, 3.9999961853027344, 4.76837158203125e-07, array([ 2.        , -1.99999809])]
# [2, 3.9999847412327654, 3.9999923706091067, 4.76837158203125e-07, array([ 2.        , -1.99999619])]
# [3, 3.999977111870976, 3.999988555919116, 4.76837158203125e-07, array([ 2.        , -1.99999428])]
# [4, 3.9999694825237384, 3.9999847412327636, 4.76837158203125e-07, array([ 2.        , -1.99999237])]
# [5, 3.9999618531910524, 3.999980926550048, 4.76837158203125e-07, array([ 2.        , -1.99999046])]
# [6, 3.9999542238729187, 3.99997711187097, 4.76837158203125e-07, array([ 2.        , -1.99998856])]
# [7, 3.9999465945693364, 3.999973297195529, 4.76837158203125e-07, array([ 2.        , -1.99998665])]
# [8, 3.999938965280306, 3.999969482523724, 4.76837158203125e-07, array([ 2.        , -1.99998474])]
# [9, 3.9999313360058273, 3.999965667855556, 4.76837158203125e-07, array([ 2.        , -1.99998283])]
# [10, 3.9999237067459004, 3.9999618531910253, 4.76837158203125e-07, array([ 2.        , -1.99998093])]
# [996, 3.9924083327541955, 3.9962023362304984, 4.76837158203125e-07, array([ 2.        , -1.99810118])]
# [997, 3.992400717843356, 3.996198525072439, 4.76837158203125e-07, array([ 2.        , -1.99809928])]
# [998, 3.9923931029470405, 3.996194713917849, 4.76837158203125e-07, array([ 2.        , -1.99809737])]
# [999, 3.992385488065251, 3.9961909027667275, 4.76837158203125e-07, array([ 2.        , -1.99809547])]
# [1000, 3.992377873197985, 3.996187091619075, 4.76837158203125e-07, array([ 2.        , -1.99809356])]
