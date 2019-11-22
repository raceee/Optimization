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
    a = (400 * num1 ** 3) - (400 * num1 * num2) + (2 * num1) - 2
    b = (200 * num2) - (200 * num1 ** 2)
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
        # print(n)
    return alpha

def modified_newton(x0, epsilon=10 ** -8, N=1000, Max_LS_iter=20, mu=10 ** -4, v=0.9, y=10 ** -4):
    data = [['iteration', 'function at x_k', 'norm of gradient', 'step size', 'x_k']]
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
        print("eig =", eig_values)
        print(np.less_equal(eig_values,0).any())
        if np.less_equal(eig_values,0).any() =='True':
            PD_adjust = abs(np.amin(eig_values)) + 0.01
            print("PD_adjust =", PD_adjust)
            h = np.add(h, (PD_adjust * np.identity(3, dtype=float))) #acceptable dimensions
            print("h = ", h)
        p_k = LA.solve(h,(-1 * gradient(x_k[0],x_k[1])))
        print("p_k = ", p_k)
        alpha = backtracking(mu, p_k, x_k)
        x_k = x_k + alpha * p_k
        k += 1
        h = hessian(x_k[0], x_k[1])
        data.append([k, function(x_k[0],x_k[1]), LA.norm(gradient(x_k[0],x_k[1])), alpha, x_k])
    return data

x0 = np.array([-1.2,1])
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
# ['iteration', 'function at x_k', 'norm of gradient', 'step size', 'x_k']
# [1, -102.51122347661024, 232.8239070897782, 4.76837158203125e-07, array([-1.19999914,  1.00008186])]
# [2, -102.50244854199502, 232.7801347977167, 4.76837158203125e-07, array([-1.19999828,  1.0001637 ])]
# [3, -102.49367519585083, 232.73637087644263, 4.76837158203125e-07, array([-1.19999742,  1.00024553])]
# [4, -102.48490343787421, 232.692615324357, 4.76837158203125e-07, array([-1.19999656,  1.00032734])]
# [5, -102.47613326776164, 232.64886813986084, 4.76837158203125e-07, array([-1.1999957 ,  1.00040914])]
# [6, -102.46736468520967, 232.60512932135538, 4.76837158203125e-07, array([-1.19999484,  1.00049092])]
# [7, -102.4585976899151, 232.5613988672426, 4.76837158203125e-07, array([-1.19999398,  1.00057268])]
# [8, -102.44983228157454, 232.51767677592412, 4.76837158203125e-07, array([-1.19999312,  1.00065443])]
# [9, -102.44106845988487, 232.4739630458027, 4.76837158203125e-07, array([-1.19999226,  1.00073616])]
# [10, -102.43230622454286, 232.43025767528084, 4.76837158203125e-07, array([-1.1999914 ,  1.00081787])]
# [996, -94.51827952924647, 193.16012398797506, 4.76837158203125e-07, array([-1.19912967,  1.07404633])]
# [997, -94.51094411393781, 193.12393746910286, 4.76837158203125e-07, array([-1.19912879,  1.0741136 ])]
# [998, -94.50361001211917, 193.08775787161667, 4.76837158203125e-07, array([-1.1991279 ,  1.07418086])]
# [999, -94.4962772235399, 193.05158519419422, 4.76837158203125e-07, array([-1.19912701,  1.07424811])]
# [1000, -94.4889457479492, 193.01541943551325, 4.76837158203125e-07, array([-1.19912612,  1.07431535])]