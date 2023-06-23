import gurobipy as gp
import numpy as np
from gurobipy import GRB


def subgrad_opt(
        A, b, ub, x0, x1, node=None, lambd=None,
        f=2, k=5, eps=0.005, omega=100):
    
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
    
    if node is not None:
        node.set_x_lb(x_best)
        node.set_lambd(lambd_best)

    return lb


def dual_lb(A, b, ub, x0, x1, node=None):
    c = np.sum(A, axis=0)
    db = b.astype(np.float32)[:]
    for j in x1:
        db -= A[:,j]

    md = gp.Model("dual")
    y = md.addMVar(A.shape[0], lb=0, name="y")
    md.setObjective(db @ y, GRB.MAXIMIZE)
    constr = md.addConstr(A.T @ y <= c)

    for j in x0 + x1:
        md.remove(constr[j])

    md.optimize()
    lambd = []
    for v in md.getVars():
        lambd.append(v.x)
    lambd = np.array(lambd)
    
    if node is not None:
        node.set_lambd(lambd)

    rc = (1 - lambd) @ A
    x = np.where(rc < 0, 1, 0)
    x[x0] = 0
    x[x1] = 1
    
    s = sum([c[j] for j in x1])
    assert np.isclose(db @ lambd + s, rc @ x + lambd @ b), \
        (db @ lambd + s, rc @ x + lambd @ b)
    lb = rc @ x + lambd @ b

    return lb


def lp_rel(A, b, ub, x0, x1, node):
    c = np.sum(A, axis=0)
    m = gp.Model("lp_rel")
    x = m.addMVar(A.shape[-1], lb=0, ub=1, name="x")
    m.setObjective(c @ x)
    m.addConstr(A @ x >= b)
    
    return m.getObjective().getValue()
