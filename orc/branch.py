import numpy as np

from .data_structures import Node


def branch_reduced_costs(A, b, bb, node):
    """Return two nodes such that one node has an additional
    variable j fixed to 0 and the other node has the same 
    additional variable j fixed to 1. 
    
    Variable j is selected as the variable with the minimum 
    reduced cost and a non-zero coefficient in the row with the 
    largest violation given the partial solution of the 
    father node.
    
    Parameters
    ----------
    A : np.ndarray
        Matrix of the left-hand side of the problem.

    b : np.ndarray
        Array of the right-hand side of the problem.

    bb : BranchAndBound
        Branch-and-bound data structure.

    node : Node
        Father node from which two children are generated.

    Returns
    -------
    nodes : list of Node
        The two children nodes.
    """
    x0 = node.get_x0()
    x1 = node.get_x1()
    x_partial = np.zeros((A.shape[-1],))
    x_partial[node.get_x1()] = 1
    rc = (1 - node.get_lambd()) @ A
    
    # Pick the row with the largest violation given the
    # solution obtained by completing the partial solution
    # of the node by fixing all the remaining variables
    # to 0.
    violation = b - (A @ x_partial)
    r = np.argmax(violation)
    
    # Columns with fixed values or with a zero entry
    # on the picked row are not candidates for branching.
    zero_entry = set(np.where(A[r] == 0)[0])
    not_candidates = list(
        set(x0).union(set(x1)).union(zero_entry))
    
    # Pick the column with minimum reduced cost
    rc[not_candidates] = np.inf
    j = np.argmin(rc)
    assert not j in not_candidates, (r, rc, x0, x1, A, b)

    return [
        Node(id=bb.get_new_id(),
             level=node.get_level() + 1, 
             x0=x0 + [j], 
             x1=x1, 
             branch_strategy=branch_reduced_costs, 
             lb_strategy=node.get_lb_strategy()),
        Node(id=bb.get_new_id(),
             level=node.get_level() + 1, 
             x0=x0, 
             x1=x1 + [j], 
             branch_strategy=branch_reduced_costs, 
             lb_strategy=node.get_lb_strategy())    
    ]


def branch_cost(A, b, bb, node):
    """Return two nodes such that one node has an additional
    variable j fixed to 0 and the other node has the same 
    additional variable j fixed to 1. 
    
    Variable j is selected as the variable with the minimum 
    cost and a non-zero coefficient in the row with the largest 
    violation given the partial solution of the father node.
    
    Parameters
    ----------
    A : np.ndarray
        Matrix of the left-hand side of the problem.

    b : np.ndarray
        Array of the right-hand side of the problem.

    bb : BranchAndBound
        Branch-and-bound data structure.

    node : Node
        Father node from which two children are generated.

    Returns
    -------
    nodes : list of Node
        The two children nodes.
    """
    x0 = node.get_x0()
    x1 = node.get_x1()
    x_partial = np.zeros((A.shape[-1],))
    x_partial[node.get_x1()] = 1
    c = np.sum(A, axis=0)
    
    # Pick the row with the largest violation given the
    # solution obtained by completing the partial solution
    # of the node by fixing all the remaining variables
    # to 0.
    violation = b - (A @ x_partial)
    r = np.argmax(violation)
    
    # Columns with fixed values or with a zero entry
    # on the picked row are not candidates for branching.
    zero_entry = set(np.where(A[r] == 0)[0])
    not_candidates = list(
        set(x0).union(set(x1)).union(zero_entry))
    
    # Pick the column with minimum cost
    c[not_candidates] = np.inf
    j = np.argmin(c)
    assert not j in not_candidates, (r, c, x0, x1, A, b)

    return [
        Node(id=bb.get_new_id(),
             level=node.get_level() + 1, 
             x0=x0 + [j], 
             x1=x1, 
             branch_strategy=branch_cost, 
             lb_strategy=node.get_lb_strategy()),
        Node(id=bb.get_new_id(),
             level=node.get_level() + 1, 
             x0=x0, 
             x1=x1 + [j], 
             branch_strategy=branch_cost, 
             lb_strategy=node.get_lb_strategy())    
    ]


def branch_beasley(A, b, bb, node):
    """Return two nodes such that one node has an additional
    variable j fixed to 0 and the other node has the same 
    additional variable j fixed to 1. 
    
    Variable j is selected as the variable with the minimum 
    reduced cost and a non-zero coefficient in the row 
    corresponding to the Lagrangean multiplier with the 
    largest value. This is the branching rule employed in [1].
    
    Parameters
    ----------
    A : np.ndarray
        Matrix of the left-hand side of the problem.

    b : np.ndarray
        Array of the right-hand side of the problem.

    bb : BranchAndBound
        Branch-and-bound data structure.

    node : Node
        Father node from which two children are generated.

    Returns
    -------
    nodes : list of Node
        The two children nodes.

    References
    ----------
    [1] J.E. Beasley, An algorithm for set covering problem, 
    European Journal of Operational Research, Volume 31, 
    Issue 1, 1987, Pages 85-93.
    """
    x0 = node.get_x0()
    x1 = node.get_x1()
    lambd = node.get_lambd()
    x_partial = np.zeros((A.shape[-1],))
    x_partial[node.get_x1()] = 1
    rc = (1 - lambd) @ A

    # Pick the uncovered row corresponding to the Lagrangean
    # multiplier with the largest value.
    uncovered_rows = np.where(b > (A @ x_partial))[0]
    r = -1
    lambd_max = -np.inf
    for i in uncovered_rows:
        if lambd[i] > lambd_max:
            lambd_max = lambd[i]
            r = i
    
    # Columns with fixed values or with a zero entry
    # on the picked row are not candidates for branching.
    zero_entry = set(np.where(A[r] == 0)[0])
    not_candidates = list(
        set(x0).union(set(x1)).union(zero_entry))
    
    # Pick the column with minimum reduced cost
    rc[not_candidates] = np.inf
    j = np.argmin(rc)
    assert not j in not_candidates, (r, rc, x0, x1, A, b)

    return [
        Node(id=bb.get_new_id(),
             level=node.get_level() + 1, 
             x0=x0 + [j], 
             x1=x1, 
             branch_strategy=branch_beasley, 
             lb_strategy=node.get_lb_strategy()),
        Node(id=bb.get_new_id(),
             level=node.get_level() + 1, 
             x0=x0, 
             x1=x1 + [j], 
             branch_strategy=branch_beasley, 
             lb_strategy=node.get_lb_strategy())    
    ]
