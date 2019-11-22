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

def quasi(x0, epsilon = 10 ** -8, N = 1000, Max_LS_iter = 20, mu = 10 ** -4, v = 0.9, y = 10 ** -4):
    k = 0
    x_k = x0
    b_k = np.identity(2, dtype="float") #acceptable dimensions
    data = [['iteration', 'norm of gradient', 'step size', 'x_k']]
    while k <= N:
        p_k = LA.solve(b_k, (-1 * gradient(x_k[0], x_k[1])))
        alpha = backtracking(mu, p_k, x_k, Max_LS_iter=20)
        mag_and_direc = alpha * p_k
        x_k1 = np.add(x_k, mag_and_direc)
        s_k = np.subtract(x_k1, x_k)
        y_k = np.subtract(gradient(x_k1[0], x_k1[1]), gradient(x_k[0], x_k[1]))
        numerator = np.matmul(np.matmul(b_k, s_k), np.matmul(b_k, s_k).transpose())
        denominator = np.matmul(s_k.transpose(), np.matmul(b_k, s_k))
        a = numerator / denominator
        b = np.matmul(y_k, y_k.transpose()) / np.matmul(y_k.transpose(), s_k)
        ab = a + b
        b_k = np.subtract(b_k, (ab * np.identity(2, dtype="float")))
        x_k = x_k1
        k += 1
        data.append([k, LA.norm(gradient(x_k[0], x_k[1])), alpha, x_k])
    return data

x0 = np.array([0,0])
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
# [1, 1.9999961853036439, 4.76837158203125e-07, array([0.00000000e+00, 9.53674316e-07])]
# [2, 1.9999969482409508, 4.76837158203125e-07, array([-1.81898940e-13,  7.62939817e-07])]
# [3, 1.9999977111788394, 4.76837158203125e-07, array([-3.27418224e-13,  5.72205208e-07])]
# [4, 1.9999984741171357, 4.76837158203125e-07, array([-4.36557786e-13,  3.81470534e-07])]
# [5, 1.9999992370558395, 4.76837158203125e-07, array([-5.09317608e-13,  1.90735795e-07])]
# [6, 1.9999999999949505, 4.76837158203125e-07, array([-5.45697673e-13,  9.89542078e-13])]
# [7, 2.000000762934469, 4.76837158203125e-07, array([-5.45697966e-13, -1.90733881e-07])]
# [8, 2.000001525874395, 4.76837158203125e-07, array([-5.09318469e-13, -3.81468817e-07])]
# [9, 2.0000022888147284, 4.76837158203125e-07, array([-4.36559167e-13, -5.72203819e-07])]
# [10, 2.0000030517554697, 4.76837158203125e-07, array([-3.27420041e-13, -7.62938886e-07])]
# [996, 2.000755509621199, 4.76837158203125e-07, array([ 1.78120472e-08, -1.88859584e-04])]
# [997, 2.000756272964193, 4.76837158203125e-07, array([ 1.78480714e-08, -1.89050384e-04])]
# [998, 2.0007570363075957, 4.76837158203125e-07, array([ 1.78841319e-08, -1.89241183e-04])]
# [999, 2.000757799651405, 4.76837158203125e-07, array([ 1.79202289e-08, -1.89431983e-04])]
# [1000, 2.0007585629956224, 4.76837158203125e-07, array([ 1.79563622e-08, -1.89622783e-04])]