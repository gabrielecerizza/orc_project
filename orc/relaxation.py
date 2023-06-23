import gurobipy as gp
import numpy as np
from gurobipy import GRB


def subgrad_opt(
        A, b, ub, x0, x1, node, lambd=None,
        f=2, k=5, eps=0.005, omega=500):
    
    lambd = np.zeros(A.shape[0]) if lambd is None else lambd
    lb = -np.inf

    unchanged = 0
    t = 0
    lambd_best = lambd
    x_best = None
    while (ub > lb):
        rc = (1 - lambd) @ A
        x = np.where(rc < 0, 1, 0)
        x[x0] = 0
        x[x1] = 1
        L = rc @ x + lambd @ b
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
    
    node.set_x_lb(x_best)
    node.set_lambd(lambd_best)

    return lb


def dual_lb(A, b, ub, x0, x1, node,):
    c = np.sum(A, axis=0)
    md = gp.Model("dual")
    y = md.addMVar(A.shape[0], lb=0, name="y")
    md.setObjective(b @ y, GRB.MAXIMIZE)
    constr = md.addConstr(A.T @ y <= c)

    for j in x0 + x1:
        md.remove(constr[j])

    try:
        md.optimize()
        lambd = []
        for v in md.getVars():
            lambd.append(v.x)
        lambd = np.array(lambd)
    except:
        lambd = None
    
    lambd = dual_feas(lambd, A)
    node.set_lambd(lambd)

    """
    rc = (1 - lambd) @ A
    x = np.where(rc < 0, 1, 0)
    x[x0] = 0
    x[x1] = 1
    """
    lb = b @ lambd

    return lb



def dual_feas(lambd, A):
    u = lambd
    s = (1 - lambd) @ A
    print("s", s)
    for j in np.where(s < 0)[0]:
        print("j", j)
        for i in np.where(A.T[j] > 0)[0]:
            print("i", i)
            if s[j] < u[i]:
                for l in np.where(A[i] > 0)[0]:
                    s[l] += u[i]
                u[i] = 0
            else:
                u[i] += s[j]
                for l in np.where(A[i] > 0)[0]:
                    s[l] -= s[j]
    print("lambd", lambd)
    print("u", u)
    return u


def dual_ascent(A, b, ub, x0, x1, node,):
    c = np.sum(A, axis=0)
    s = c
    lambd = np.zeros((A.shape[0]))
    N = []
    for i in range(A.shape[0]):
        idx = np.where(A[i] > 0)[0]
        N.append((i, idx))

    list.sort(N, key=lambda x: x[0])

    for i, _ in N:
        lambd[i] = min([s[j] for j in N[i][1]])
        for j in range(A.shape[-1]):
            if j in N[i][1]:
                s[j] -= lambd[i]
    node.set_lambd(lambd)

    rc = (1 - lambd) @ A
    x = np.where(rc < 0, 1, 0)
    x[x0] = 0
    x[x1] = 1
    lb = c @ x + lambd @ b

    return b @ lambd


def dual_ascent2(A, b, ub, x0, x1, node,):
    c = np.sum(A, axis=0)
    lambd = np.zeros((A.shape[0]))
    s = (1 - lambd) @ A
    N = []
    for i in range(A.shape[0]):
        idx = np.where(A[i] > 0)[0]
        N.append((i, idx))

    list.sort(N, key=lambda x: x[0], reverse=True)

    for i, _ in N:
        lambd[i] = max(0, lambd[i] + min(0, min([s[j] for j in N[i][1]])))

    list.sort(N, key=lambda x: x[0])

    for i, _ in N:
        lambd[i] = lambd[i] + min([s[j] for j in N[i][1]])

    lambd
    node.set_lambd(lambd)

    rc = (1 - lambd) @ A
    x = np.where(rc < 0, 1, 0)
    x[x0] = 0
    x[x1] = 1
    lb = c @ x + lambd @ b

    return b @ lambd




def lp_rel(A, b, ub, x0, x1, node):
    # TODO: use LP relaxation
    c = np.sum(A, axis=0)
    m = gp.Model("lp_rel")
    x = m.addMVar(A.shape[-1], lb=0, name="x")
    m.setObjective(c @ x)
    m.addConstr(A @ x >= b)
    
    return m.getObjective().getValue()