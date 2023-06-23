import numpy as np

from .primal import hall_hochbaum
from .branch_bound import Node
from .reduction import column_inclusion, lagr_penalties


class BBCallback:
    def on_preprocess(self, bb, A, b):
        pass

    def on_reduction(self, node, A, ub):
        pass


class SingleReductionCallback(BBCallback):
    def on_preprocess(self, bb, A, b):
        primal_heur(bb, A, b)

    def on_reduction(self, node, A, ub):
        # lagr_penalties(node, A, ub)
        pass

class ColumnInclusionCallback(BBCallback):
    def on_reduction(self, node, A, ub):
        column_inclusion(node, A, ub)


def primal_heur(bb, A, b):
    x = hall_hochbaum(A, b)
    ub = np.sum(A, axis=0) @ x
    bb.set_ub(ub)
    bb.set_x_best(x)

    # Add a fictitious best node corresponding to the
    # primal heuristic solution.
    x0=np.where(x == 0)[0]
    x1=np.where(x == 1)[0]
    n = Node(id=-1, level=-1, x0=x0, x1=x1, 
             branch_strategy=None, lb_strategy=None)
    bb.set_best(n)
