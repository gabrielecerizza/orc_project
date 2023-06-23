import numpy as np


def lagr_penalties(node, A, ub):
    lb = node.get_lb()
    rc = (1 - node.get_lambd()) @ A
    fix0, fix1 = [], []

    for j in range(A.shape[-1]):
        if rc[j] >= 0 and lb + rc[j] > ub \
            and not j in node.get_x1():
            fix0.append(j)
        elif rc[j] < 0 and lb - rc[j] > ub \
            and not j in node.get_x0():
            fix1.append(j)

    node.add_to_x0(fix0)
    node.add_to_x1(fix1)


def column_inclusion(node, A, b):
    x = np.zeros((A.shape[-1],))
    x[node.get_x1()] = 1
    unassigned = [j for j in range(A.shape[-1]) 
                  if j not in node.get_x1() + node.get_x0()]
    fix1 = []

    uncovered = np.where(A @ x < b)[0]
    for r in uncovered:
        nonzero = np.where(A[r] > 0)[0]
        for j in unassigned:
            if j in nonzero and len(nonzero) == 1:
                fix1.append(j)

    node.add_to_x1(fix1)
