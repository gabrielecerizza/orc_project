import numpy as np


def lagr_penalties(node, A, ub):
    """Return true if the size of the problem was reduced 
    by fixing variables to 0 and 1.

    This method fixes to 0 those variables whose positive 
    reduced costs, when added to the lower bound of the node, 
    would exceed the incumbent upper bound; and fixes to 1 
    those variables whose negative reduced costs, when
    subtracted from the lower bound of the node, would
    exceed the incumbent upper bound. This method is taken 
    from [1].

    Parameters
    ----------
    node : Node
        Current node of the branch-and-bound data structure.
    
    A : np.ndarray
        Matrix of the left-hand side of the problem.

    ub : int
        Value of the incumbent upper bound.

    Returns
    -------
    reduced : bool
        True if at least a variable has been fixed to 0 or 1.
    
    References
    ----------
    [1] J.E. Beasley, An algorithm for set covering problem, 
    European Journal of Operational Research, Volume 31, 
    Issue 1, 1987, Pages 85-93.
    """
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

    return len(fix0) > 0 or len(fix1) > 0


def column_inclusion(node, A, b):
    """Return true if the size of the problem was reduced 
    by fixing variables to 1.

    When a row can be covered only if all the unassigned
    variables are fixed to 1, feasibility requires to fix 
    those variables to 1. This method is adapted from [1].

    Parameters
    ----------
    node : Node
        Current node of the branch-and-bound data structure.
    
    A : np.ndarray
        Matrix of the left-hand side of the problem.

    b : np.ndarray
        Array of the right-hand side of the problem.

    Returns
    -------
    reduced : bool
        True if at least a variable has been fixed to 1.
    
    References
    ----------
    [1] J.E. Beasley, An algorithm for set covering problem, 
    European Journal of Operational Research, Volume 31, 
    Issue 1, 1987, Pages 85-93.
    """
    x = np.zeros((A.shape[-1],))
    x[node.get_x1()] = 1
    unassigned = [j for j in range(A.shape[-1]) 
                  if j not in node.get_x1() + node.get_x0()]
    fix1 = []

    uncovered = np.where(A @ x < b)[0]
    for r in uncovered:
        amount_covered = sum([A[r][j] for j in node.get_x1()])
        amount_left = b[r] - amount_covered
        amount_coverable = sum([A[r][j] for j in unassigned])
        assert amount_coverable >= amount_left, \
            (A[r], b[r], unassigned)

        for j in unassigned:
            if amount_coverable - A[r][j] < amount_left:
                fix1.append(j)

    node.add_to_x1(fix1)

    return len(fix1) > 0
