import numpy as np


def subgrad_opt(
        A, b, ub, X0, X1,
        lambd=None, lb=None, f=2, k=15, eps=0.005, omega=500):
    if lambd is None:
        lambd = np.zeros(A.shape[0])
    if lb is None:
        lb = -np.inf

    unchanged = 0
    t = 0
    lambd_best = lambd
    x_best = None
    while (ub > lb):
        c = (1 - lambd) @ A
        x = np.where(c < 0, 1, 0)
        x[X0] = 0
        x[X1] = 1
        L = c @ x + lambd @ b
        g = b - A @ x
        if L > lb:
            lb = L
            lambd_best = lambd
            x_best = x
            unchanged = 0
        else:
            unchanged += 1
        if unchanged == k:
            unchanged = 0
            f /= 2
        sigma = f * (ub - lb) / np.linalg.norm(g) ** 2
        lambd = np.maximum(
            np.zeros_like(lambd), lambd + sigma * g)
        t += 1
        if f < eps or t > omega:
            break
    
    return lambd_best, lb, x_best
