import numpy as np

from .branch_bound import Node

EPS = 1e-6


def _check_feasibility(A, b, x):
        assert np.all(A @ x >= b), (A @ x, b)


def _build_solution(A, cover):
    x = np.zeros((A.shape[-1],))
    x[cover] = 1
    return x


def _find_cover(A, b, method):
    cover = _inner_find_cover(np.copy(A), np.copy(b), method)
    x = _build_solution(A, cover)
    _check_feasibility(A, b, x)
    return x


def _inner_find_cover(A, b, method):
    cover = []
    c = np.sum(A, axis=0)
    match method:
        case "greedy":
            while np.any(b > 0):
            
                # Pick the row that is easiest to cover,
                # since it has the largest ratio between the
                # sum of the coefficients on the left hand 
                # side and b.
                M = [sum(A[i]) / (b_i or EPS)
                     for i, b_i in enumerate(b)]
                i = np.argmax(M)
                
                # Pick the column that covers the most.
                j = np.argmax(A[i])
                
                cover.append(j)
                b = b - A[:,j]
                A[:,j] -= A[:,j]

        case "dobson":
            while np.any(b > 0):
                A = np.minimum(A.T, b).T

                # To avoid division by zero, we copy
                # A and replace zero with a small epsilon
                A2 = np.copy(A).astype(np.float32)
                A2[A2 == 0] = EPS

                # c is already the sum of A over axis 0,
                # so, unless some entries of A are
                # decreased in the first step of the while
                # loop, all non-zero columns will have the
                # same ratio. 
                k = np.argmin(c / np.sum(A2, axis=0))
                
                cover.append(k)
                b = b - A[:,k]
                A[:,k] -= A[:,k]

        case "hall_hochbaum":
            while np.any(b > 0):
                rsum = np.sum(A, axis=-1)
                space = rsum - b
                
                f = [1 / (c[j] or EPS) * 
                     sum([(b[i] * A[i][j]) / (space[i]) 
                        for i in np.where(b > 0)[0]]) 
                        for j in range(A.shape[-1])]

                k = np.argmax(f)
                cover.append(k)
                b = b - A[:,k]
                A[:,k] -= A[:,k]

    return cover


def greedy(A, b):
    return _find_cover(A, b, "greedy")


def dobson(A, b):
    return _find_cover(A, b, "dobson")


def hall_hochbaum(A, b):
    return _find_cover(A, b, "hall_hochbaum")


#TODO: hall_hochbaum gets better than dobson the bigger the problem