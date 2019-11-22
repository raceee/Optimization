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
        print("alpha = ", alpha)
        n += 1
    return alpha

def quasi(x0, epsilon = 10 ** -8, N = 1000, Max_LS_iter = 20, mu = 10 ** -4, v = 0.9, y = 10 ** -4):
    k = 0
    x_k = x0
    b_k = np.identity(3, dtype="float") #acceptable dimensions
    data = [['iteration','function at x_k', 'norm of gradient', 'step size', 'x_k']]
    for i in range(N):
        p_k = LA.solve(b_k, (-1 * gradient(x_k[0], x_k[1], x_k[2])))
        alpha = backtracking(mu, p_k, x_k, Max_LS_iter=20)
        mag_and_direc = alpha * p_k
        x_k1 = np.add(x_k, mag_and_direc)
        s_k = np.subtract(x_k1, x_k)
        y_k = np.subtract(gradient(x_k1[0], x_k1[1], x_k1[2]), gradient(x_k[0], x_k[1], x_k[2]))
        numerator = np.matmul(np.matmul(b_k, s_k), np.matmul(b_k, s_k).transpose())
        denominator = np.matmul(s_k.transpose(), np.matmul(b_k, s_k))
        a = numerator / denominator
        b = np.matmul(y_k, y_k.transpose()) / np.matmul(y_k.transpose(), s_k)
        ab = a + b
        b_k = np.subtract(b_k, (ab * np.identity(3, dtype="float")))
        x_k = x_k1
        k = i
        data.append([k, function(x_k[0],x_k[1],x_k[2]),LA.norm(gradient(x_k[0], x_k[1], x_k[2])), alpha, x_k])
    return data

x0 = np.array([1,1,1])
data = quasi(x0)
if len(data) > 15:
    a = data[0:11]
    a.extend(data[len(data) - 6:len(data) - 1])
    for item in a:
        print(item)
else:
    for item in data:
        print(item)

# ['iteration', 'function at x_k', 'norm of gradient', 'step size', 'x_k']
# [0, 2.99999427795683, 3.464098311513015, 4.76837158203125e-07, array([0.99999905, 0.99999905, 0.99999905])]
# [1, 2.9999971389750044, 3.464099963323809, 4.76837158203125e-07, array([0.99999952, 0.99999952, 0.99999952])]
# [2, 2.9999999999959073, 3.464101615135392, 4.76837158203125e-07, array([1., 1., 1.])]
# [3, 3.0000028610195386, 3.4641032669477614, 4.76837158203125e-07, array([1.00000048, 1.00000048, 1.00000048])]
# [4, 3.0000057220458984, 3.464104918760919, 4.76837158203125e-07, array([1.00000095, 1.00000095, 1.00000095])]
# [5, 3.0000085830749867, 3.4641065705748644, 4.76837158203125e-07, array([1.00000143, 1.00000143, 1.00000143])]
# [6, 3.0000114441068035, 3.464108222389597, 4.76837158203125e-07, array([1.00000191, 1.00000191, 1.00000191])]
# [7, 3.000014305141349, 3.4641098742051177, 4.76837158203125e-07, array([1.00000238, 1.00000238, 1.00000238])]
# [8, 3.0000171661786226, 3.4641115260214255, 4.76837158203125e-07, array([1.00000286, 1.00000286, 1.00000286])]
# [9, 3.000020027218625, 3.4641131778385215, 4.76837158203125e-07, array([1.00000334, 1.00000334, 1.00000334])]
# [994, 3.0028394770091653, 3.465740600223372, 4.76837158203125e-07, array([1.00047313, 1.00047313, 1.00047313])]
# [995, 3.002842340740733, 3.465742252817271, 4.76837158203125e-07, array([1.00047361, 1.00047361, 1.00047361])]
# [996, 3.0028452044750336, 3.4657439054119585, 4.76837158203125e-07, array([1.00047409, 1.00047409, 1.00047409])]
# [997, 3.0028480682120637, 3.4657455580074332, 4.76837158203125e-07, array([1.00047457, 1.00047457, 1.00047457])]
# [998, 3.0028509319518246, 3.4657472106036957, 4.76837158203125e-07, array([1.00047504, 1.00047504, 1.00047504])]