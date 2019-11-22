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
# x_k[0], x_k[1], x_k[2]
def gradient_descent(x0, epsilon=10 ** -8, N=1000, Max_LS_iter=20, mu=10 ** -4, v=0.9, y=10 ** -4):
    data = [['iteration', 'norm of gradient', 'step size', 'x_k']]
    k = 0
    x_k = x0
    while (LA.norm(gradient(x_k[0], x_k[1], x_k[2]),2) / (1 + abs(function(x_k[0], x_k[1], x_k[2])))) > epsilon and k <= N:  #figure out gradient information on how to plug stuff into it
        p_k = -gradient(x_k[0], x_k[1], x_k[2])
        alpha = backtracking(mu, p_k, x_k)
        x_k = x_k + (alpha * p_k)
        k += 1
        data.append([k, LA.norm(gradient(x_k[0], x_k[1], x_k[2])), alpha, x_k])
    if k > N:
        print("Number of iterations exceeded limit: ", N)
        return data
    else:
        return data

x0 = np.array([1,1,1])
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
# [1, 3.464098311513015, 4.76837158203125e-07, array([0.99999905, 0.99999905, 0.99999905])]
# [2, 3.4640950078914257, 4.76837158203125e-07, array([0.99999809, 0.99999809, 0.99999809])]
# [3, 3.464091704272987, 4.76837158203125e-07, array([0.99999714, 0.99999714, 0.99999714])]
# [4, 3.464088400657699, 4.76837158203125e-07, array([0.99999619, 0.99999619, 0.99999619])]
# [5, 3.4640850970455612, 4.76837158203125e-07, array([0.99999523, 0.99999523, 0.99999523])]
# [6, 3.464081793436575, 4.76837158203125e-07, array([0.99999428, 0.99999428, 0.99999428])]
# [7, 3.4640784898307384, 4.76837158203125e-07, array([0.99999332, 0.99999332, 0.99999332])]
# [8, 3.4640751862280528, 4.76837158203125e-07, array([0.99999237, 0.99999237, 0.99999237])]
# [9, 3.4640718826285175, 4.76837158203125e-07, array([0.99999142, 0.99999142, 0.99999142])]
# [10, 3.464068579032133, 4.76837158203125e-07, array([0.99999046, 0.99999046, 0.99999046])]
# [996, 3.4608127655486727, 4.76837158203125e-07, array([0.99905059, 0.99905059, 0.99905059])]
# [997, 3.460809465060424, 4.76837158203125e-07, array([0.99904964, 0.99904964, 0.99904964])]
# [998, 3.4608061645753234, 4.76837158203125e-07, array([0.99904869, 0.99904869, 0.99904869])]
# [999, 3.4608028640933703, 4.76837158203125e-07, array([0.99904773, 0.99904773, 0.99904773])]
# [1000, 3.4607995636145645, 4.76837158203125e-07, array([0.99904678, 0.99904678, 0.99904678])]