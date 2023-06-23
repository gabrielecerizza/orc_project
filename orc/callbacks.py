import numpy as np

from .primal import hall_hochbaum, primal_heur
from .branch_bound import Node
from .reduction import column_inclusion, lagr_penalties


class BBCallback:
    def on_preprocess(self, bb, A, b, node):
        pass

    def on_reduction(self, node, A, ub):
        pass


class PrimalHeurCallback(BBCallback):
    def __init__(self, step=1, only_root=False):
        self.step = step
        self.only_root = only_root

    def on_preprocess(self, bb, A, b, node):
        if self.only_root:
            if node.get_level() == 0:
                primal_heur(bb, A, b, bb.get_ub(), node)
        elif bb.get_node_count() % self.step == 0:
            primal_heur(bb, A, b, bb.get_ub(), node)


class LagrPenaltiesReductionCallback(BBCallback):
    def on_reduction(self, node, A, ub):
        lagr_penalties(node, A, ub)


class ColumnInclusionCallback(BBCallback):
    def on_reduction(self, node, A, ub):
        column_inclusion(node, A, ub)
