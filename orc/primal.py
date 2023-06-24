import numpy as np

from .data_structures import Node

EPS = 1e-6


def _check_feasibility(A, b, x, method):
        assert np.all(A @ x >= b), (method, A @ x, b, x)


def _build_solution(A, cover):
    x = np.zeros((A.shape[-1],))
    x[cover] = 1
    return x


def _find_cover(A, b, x0, x1, method):
    cover = _inner_find_cover(np.copy(A), 
                              np.copy(b), x0, x1, method)
    x = _build_solution(A, cover)
    _check_feasibility(A, b, x, method)
    return x


def _inner_find_cover(A, b, x0, x1, method):
    cover = [] + x1
    b = b.astype(np.float32)
    
    # Adjust the values of A and b according
    # to the variables that are already fixed.
    for j in x1:
        b -= A[:,j]
        A[:,j] -= A[:,j]
    for j in x0:
        A[:,j] -= A[:,j]
    
    c = np.sum(A, axis=0)
    match method:
        case "greedy":
            while np.any(b > 0):
            
                # Pick the row that is easiest to cover,
                # since it has the largest ratio between the
                # sum of the coefficients on the left-hand 
                # side and b.
                M = [sum(A[i]) / b_i if b_i > 0 else -1
                     for i, b_i in enumerate(b)]
                i = np.argmax(M)
                
                # Pick the column that covers the most and
                # that is not fixed.
                a_max = -np.inf
                j = -1
                for k in range(A.shape[-1]):
                    if k not in x0 + x1:
                        if A[i][k] > a_max:
                            a_max = A[i][k]
                            j = k
                
                assert j > -1
                cover.append(j)
                x1 += [j]
                b -= A[:,j]
                A[:,j] -= A[:,j]

        case "dobson":
            while np.any(b > 0):
                A = np.minimum(A.T, b).T

                # To avoid division by zero, we copy
                # A and replace zero with a small epsilon.
                A2 = np.copy(A).astype(np.float32)
                A2[A2 == 0] = EPS

                # c is already the sum of A over axis 0,
                # so, unless some entries of A are
                # decreased in the first step of the while
                # loop, all non-zero columns will have the
                # same ratio.
                f = c / np.sum(A2, axis=0)
                f_min = np.inf
                k = -1
                for j in range(A.shape[-1]):
                    if j not in x0 + x1:
                        if f[j] < f_min:
                            f_min = f[j]
                            k = j 
                
                assert k > -1
                cover.append(k)
                x1 += [k]
                b -= A[:,k]
                A[:,k] -= A[:,k]

        case "hall_hochbaum":
            while np.any(b > 0):
                rsum = np.sum(A, axis=-1)
                space = rsum - b
                
                f = [1 / (c[j] or EPS) * 
                     sum([(b[i] * A[i][j]) / (space[i] or EPS) 
                        for i in np.where(b > 0)[0]]) 
                        for j in range(A.shape[-1])]

                f_max = -np.inf
                k = -1
                for j in range(A.shape[-1]):
                    if j not in x0 + x1:
                        if f[j] > f_max:
                            f_max = f[j]
                            k = j
                
                assert k > -1
                cover.append(k)
                x1 += [k]
                b -= A[:,k]
                A[:,k] -= A[:,k]

    return cover


def greedy(A, b, x0, x1):
    """Return a feasible solution for the problem described
    by A and b, keeping into account the variables already
    fixed to 0 and to 1.

    This method selects greedily the variables of the cover,
    picking the row with the large ratio between left-hand
    side and right-hand side and then picking the column
    with the largest coefficient.

    Parameters
    ----------
    A : np.ndarray
        Matrix of the left-hand side of the problem.

    b : np.ndarray
        Array of the right-hand side of the problem.

    x0 : list of int
        Indices of variables fixed to 0 in the current node.

    x1 : list of int
        Indices of variables fixed to 1 in the current node.

    Returns
    -------
    solution : list of int
        The vector of variables fixed to 0 and 1 corresponding
        to a feasible solution.
    """
    return _find_cover(A, b, x0, x1, "greedy")


def dobson(A, b, x0, x1):
    """Return a feasible solution for the problem described
    by A and b, keeping into account the variables already
    fixed to 0 and to 1.

    This method is adapted from the article by Dobson [1].

    Parameters
    ----------
    A : np.ndarray
        Matrix of the left-hand side of the problem.

    b : np.ndarray
        Array of the right-hand side of the problem.

    x0 : list of int
        Indices of variables fixed to 0 in the current node.

    x1 : list of int
        Indices of variables fixed to 1 in the current node.

    Returns
    -------
    solution : list of int
        The vector of variables fixed to 0 and 1 corresponding
        to a feasible solution.

    References
    ----------
    [1] Dobson, Gregory. “Worst-Case Analysis of Greedy 
    Heuristics for Integer Programming with Nonnegative Data.” 
    Math. Oper. Res. 7 (1982): 515-531.
    """
    return _find_cover(A, b, x0, x1, "dobson")


def hall_hochbaum(A, b, x0, x1):
    """Return a feasible solution for the problem described
    by A and b, keeping into account the variables already
    fixed to 0 and to 1.

    This method is adapted from the article by Hall and
    Hochbaum [1].

    Parameters
    ----------
    A : np.ndarray
        Matrix of the left-hand side of the problem.

    b : np.ndarray
        Array of the right-hand side of the problem.

    x0 : list of int
        Indices of variables fixed to 0 in the current node.

    x1 : list of int
        Indices of variables fixed to 1 in the current node.

    Returns
    -------
    solution : list of int
        The vector of variables fixed to 0 and 1 corresponding
        to a feasible solution.

    References
    ----------
    [1] Nicholas G. Hall, Dorit S. Hochbaum, The multicovering 
    problem, European Journal of Operational Research, 
    Volume 62, Issue 3, 1992, Pages 323-339.
    """
    return _find_cover(A, b, x0, x1, "hall_hochbaum")


def primal_heur(bb, A, b, ub, node):
    """Determine a primal feasible solution to the problem
    described by A and b by way of an heuristic adapted
    from [1] and update the incumbent upper bound of the
    branch-and-bound data structure if needed.
    
    Parameters
    ----------
    bb : BranchAndBound
        Branch-and-bound data structure.

    A : np.ndarray
        Matrix of the left-hand side of the problem.

    b : np.ndarray
        Array of the right-hand side of the problem.

    ub : int
        Current incumbent upper bound.

    node : Node
        Current node in the branch-and-bound data structure.

    References
    ----------
    [1] Nicholas G. Hall, Dorit S. Hochbaum, The multicovering 
    problem, European Journal of Operational Research, 
    Volume 62, Issue 3, 1992, Pages 323-339.
    """
    x = hall_hochbaum(
        A, b, node.get_x0()[:], node.get_x1()[:])
    new_ub = np.sum(A, axis=0) @ x
    if new_ub < ub:
        bb.set_ub(new_ub)
        bb.set_x_best(x)

        # Add a fictitious best node corresponding to the
        # primal heuristic solution.
        x0=np.where(x == 0)[0]
        x1=np.where(x == 1)[0]
        n = Node(id=-1, level=-1, x0=x0, x1=x1, 
                 branch_strategy=None, lb_strategy=None)
        n.get_val(A)
        bb.set_best(n)
