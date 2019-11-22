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

def quasi(x0, epsilon = 10 ** -8, N = 1000, Max_LS_iter = 20, mu = 10 ** -4, v = 0.9, y = 10 ** -4):
    k = 0
    x_k = x0
    b_k = np.identity(2, dtype="float") #acceptable dimensions
    data = [['iteration', 'norm of gradient', 'step size', 'x_k']]
    while k <= N:
        p_k = LA.solve(b_k, (-1 * gradient(x_k[0], x_k[1])))
        print("p_k = ", p_k)
        alpha = backtracking(mu, p_k, x_k, Max_LS_iter=20)
        mag_and_direc = alpha * p_k
        x_k1 = np.add(x_k, mag_and_direc)
        print("x_k1 = ", x_k1)
        s_k = np.subtract(x_k1, x_k)
        print("s_k = ", s_k)
        y_k = np.subtract(gradient(x_k1[0], x_k1[1]), gradient(x_k[0], x_k[1]))
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
        data.append([k, LA.norm(gradient(x_k[0], x_k[1])), alpha, x_k])
    return data
x0 = np.array([2,-2])
data = quasi(x0)
if len(data) > 15:
    a = data[0:11]
    a.extend(data[len(data) - 6:len(data) - 1])
    for item in a:
        print(item)
else:
    for item in data:
        print(item)


# ['iteration', 'norm of gradient', 'step size', 'x_k']
# [1, 3.9999961853027344, 4.76837158203125e-07, array([ 2.        , -1.99999809])]
# [2, 3.999998092649548, 4.76837158203125e-07, array([ 2.        , -1.99999905])]
# [3, 3.9999999999972715, 4.76837158203125e-07, array([ 2., -2.])]
# [4, 4.000001907345904, 4.76837158203125e-07, array([ 2.        , -2.00000095])]
# [5, 4.000003814695447, 4.76837158203125e-07, array([ 2.        , -2.00000191])]
# [6, 4.000005722045898, 4.76837158203125e-07, array([ 2.        , -2.00000286])]
# [7, 4.00000762939726, 4.76837158203125e-07, array([ 2.        , -2.00000381])]
# [8, 4.0000095367495305, 4.76837158203125e-07, array([ 2.        , -2.00000477])]
# [9, 4.000011444102712, 4.76837158203125e-07, array([ 2.        , -2.00000572])]
# [10, 4.000013351456801, 4.76837158203125e-07, array([ 2.        , -2.00000668])]
# [996, 4.001894445222076, 4.76837158203125e-07, array([ 2.        , -2.00094722])]
# [997, 4.001896353474072, 4.76837158203125e-07, array([ 2.        , -2.00094818])]
# [998, 4.001898261726977, 4.76837158203125e-07, array([ 2.        , -2.00094913])]
# [999, 4.001900169980791, 4.76837158203125e-07, array([ 2.        , -2.00095008])]
# [1000, 4.001902078235517, 4.76837158203125e-07, array([ 2.        , -2.00095104])]