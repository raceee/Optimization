#This is the code to question 11.5.8
import numpy as np
from numpy import linalg as LA

def function1(num1, num2, num3):
    return ((num1 ** 2) + (num2 ** 2) + (num3 ** 2))
def gradient1(num1, num2, num3):
    vector = [2 * num1, 2 * num2, 2 * num3]
    vector = np.asarray(vector).transpose()
    return vector
def hessian1(num1, num2, num3):
    return 2 * np.identity(3, dtype=float)


def function2(num1, num2):
    return ((num1 ** 2) (2 * (num2 ** 2)) - (2 * num1 * num2) - (2 * num2))
def gradient2():
    a = (2 * num1) - (2 * num2)
    b = (4 * num2) - (2 * num1) - 2
    vector = [a,b]
    vector = np.asarray(vector).transpose()
    return vector
def hessian2():
    return np.array([2, -2],[-2, 4])


def function3():
    return (100 * (num2 - (num1 ** 2)**2) + (1 - num1) ** 2)
def gradient3():
    a = (400 * num1 ** 3) - (400 * num1 * num2) + (2 * num1) - 2
    b = (200 * num2) - (200 * num1 ** 2)
    vector = [a,b]
    vector = np.asarray(vector).transpose()
    return vector
def hessian3():
    a = (1200 * num1 ** 2) - (400 * num2) + 2
    b = -400 * num1
    c = -400 * num1
    d = 200
    return np.array([a,b],[c,d])


def function4():
    return (((num1 + num2) ** 4) + (num2 ** 2))
def gradient4(num1, num2):
    a = (4 * (num1 + num2) ** 3)
    b = (4 * (num1 + num2) **3) + 2 * num2
    return np.array([a,b])
def hessian4():
    a = (num1 + num2) **2
    b = (num1 + num2) **2
    c = (num1 + num2) **2
    d = (num1 + num2) **2 + (1/6)
    return np.array([a,b],[c,d])


def function5(num1, num2, c):
    return (((num1 - 1) ** 2) + ((num2 - 1)**2) + c * ((num1 ** 2) + (num2 ** 2) - 0.25) ** 2)
def gradient5():
    return
def hessian5():
    return


def modified_newton(epsilon, N, Max_LS_iter, mu, v, y, x0):
    k = 0
    x_k = x0
    h = hessian(x_k)

    while LA.norm(gradient(x_k))/ (1 + abs(function(x_k)) > epsilon) and k <= N:
        #Check that hessian is positive definite
        if LA.eig(h) <= 0:
            PD_adjust = np.amin(LA.eig(h)) + 0.01
            h = h + PD_adjust * np.identity(2, dtype=float) #acceptable dimensions
        p_k = LA.solve(h,(-1 * gradient(x_k)))
        alpha = backtracking()
        x_k = x_k + alpha * p_k
        k += 1
        h = hessian(x_k)
    return x_k, function(x_k), k