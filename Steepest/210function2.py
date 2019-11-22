import numpy as np
from numpy import linalg as LA
epsilon = 10 ** -8
N = 1000
Max_LS_iter = 20
mu = 10 ** -4
v = 0.9
y = 10 ** -4


def function(num1, num2):
    return ((num1 ** 2) + (2 * (num2 ** 2)) - (2 * num1 * num2) - (2 * num2))
def gradient(num1, num2):
    a = (2 * num1) - (2 * num2)
    b = (4 * num2) - (2 * num1) - 2
    vector = [a,b]
    vector = np.asarray(vector).transpose()
    return vector
def hessian():
    a = np.array([[2, -2],[-2, 4]])
    print(type(a))
    return a


def backtracking(mu, p_k, x_k, Max_LS_iter=20):
    n = 0
    alpha = 1
    arg1 = np.add(x_k, (alpha * p_k))
    arg2 = x_k
    while function(arg1[0], arg1[1]) > (function(arg2[0], arg2[1]) + (mu * alpha) * np.matmul(np.transpose(gradient(arg2[0], arg2[1])), p_k)) and n <= Max_LS_iter:
        alpha = alpha / 2
        n += 1
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

x0 = np.array([0,0])
data = gradient_descent(x0)
if len(data) > 15:
    a = data[0:11]
    a.extend(data[len(data) - 6:len(data) - 1])
    for item in a:
        print(item)
else:
    for item in data:
        print(item)


# Number of iterations exceeded limit:  1000
# ['iteration', 'norm of gradient', 'step size', 'x_k']
# [1, 1.9999961853036439, 4.76837158203125e-07, array([0.00000000e+00, 9.53674316e-07])]
# [2, 1.9999923706182017, 4.76837158203125e-07, array([9.09494702e-13, 1.90734681e-06])]
# [3, 1.9999885559436734, 4.76837158203125e-07, array([2.72848150e-12, 2.86101749e-06])]
# [4, 1.999984741280059, 4.76837158203125e-07, array([5.45695780e-12, 3.81468635e-06])]
# [5, 1.9999809266273585, 4.76837158203125e-07, array([9.09492100e-12, 4.76835339e-06])]
# [6, 1.9999771119855718, 4.76837158203125e-07, array([1.36423685e-11, 5.72201861e-06])]
# [7, 1.9999732973546986, 4.76837158203125e-07, array([1.90992977e-11, 6.67568202e-06])]
# [8, 1.9999694827347396, 4.76837158203125e-07, array([2.54657059e-11, 7.62934360e-06])]
# [9, 1.999965668125694, 4.76837158203125e-07, array([3.27415907e-11, 8.58300336e-06])]
# [10, 1.9999618535275625, 4.76837158203125e-07, array([4.09269493e-11, 9.53666131e-06])]
# [996, 1.9962059658394737, 4.76837158203125e-07, array([4.50236782e-07, 9.48959003e-04])]
# [997, 1.9962021619996548, 4.76837158203125e-07, array([4.51141350e-07, 9.49910868e-04])]
# [998, 1.9961983581707223, 4.76837158203125e-07, array([4.52046825e-07, 9.50862731e-04])]
# [999, 1.996194554352676, 4.76837158203125e-07, array([4.52953208e-07, 9.51814592e-04])]
# [1000, 1.996190750545516, 4.76837158203125e-07, array([4.53860497e-07, 9.52766451e-04])]