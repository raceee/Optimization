import numpy as np
from numpy import linalg as LA
epsilon = 10 ** -8
N = 1000
Max_LS_iter = 20
mu = 10 ** -4
v = 0.9
y = 10 ** -4


def function(num1,num2):
    return (100 * (num2 - (num1 ** 2)**2) + (1 - num1) ** 2)
def gradient(num1,num2):
    a = (num1 ** 3) - (num1 * num2) + ((1/200) * num1) - (1/200)
    b = (num2) - (num1 ** 2)
    vector = [a,b]
    vector = np.asarray(vector).transpose()
    return vector
def hessian(num1,num2):
    a = (3 * num1 ** 2) - (num2) + 2
    b = -num1
    c = -num1
    d = 0.5
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

x0 = np.array([-1.2,1])
data = quasi(x0)
if len(data) > 15:
    a = data[0:11]
    a.extend(data[len(data) - 6:len(data) - 1])
    for item in a:
        print(item)
else:
    for item in data:
        print(item)

#I couldn't figure out what was going on here
# ['iteration', 'norm of gradient', 'step size', 'x_k']
# [1, 0.6957866626446327, 4.76837158203125e-07, array([-1.19999974,  1.00000021])]
# [2, 1.4612527099196073, 1, array([-1.32936586,  0.8943952 ])]
# [3, 3.3880532927407785, 1, array([-1.58368573,  0.70498727])]
# [4, 8.649818661197637, 1, array([-2.04929923,  0.41230514])]
# [5, 24.339602108072235, 1, array([-2.84452848,  0.02501657])]
# [6, 73.32409793819745, 1, array([-4.11137257, -0.41997235])]
# [7, 228.3629654732055, 1, array([-6.03588169, -0.88789851])]
# [8, 719.5444330647788, 1, array([-8.89123148, -1.36088722])]
# [9, 2272.774688658043, 1, array([-13.08837241,  -1.83290852])]
# [10, 7175.788217185794, 1, array([-19.23954161,  -2.30286596])]
# [996, nan, 1, array([nan, nan])]
# [997, nan, 1, array([nan, nan])]
# [998, nan, 1, array([nan, nan])]
# [999, nan, 1, array([nan, nan])]
# [1000, nan, 1, array([nan, nan])]