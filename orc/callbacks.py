from abc import ABC

from .primal import primal_heur
from .reduction import column_inclusion, lagr_penalties


class BBCallback(ABC):
    """Callback abstract class for the BranchAndBound
    data structure.
    """
    def on_run_heuristics(self, bb, A, b, node):
        """Called immediately after a node is removed 
        from the tree, before reduction and branching. 
        
        Parameters
        ----------
        bb : BranchAndBound
            Branch-and-bound data structure.

        A : np.ndarray
            Matrix of the left-hand side of the problem.

        b : np.ndarray
            Array of the right-hand side of the problem.

        node : Node
            Current node.
        """
        pass

    def on_reduction(self, node, A, b, ub):
        """Called immediately before branching. 
        
        Parameters
        ----------
        node : Node
            Current node.

        A : np.ndarray
            Matrix of the left-hand side of the problem.

        b : np.ndarray
            Array of the right-hand side of the problem.

        ub : int
            Current incumbent upper bound.

        Returns
        -------
        reduced : bool
            True if the problem has been reduced.
        """
        return False


class PrimalHeurCallback(BBCallback):
    """Callback computing a new primal heuristic upper
    bound every time a node is removed from the tree.
    """
    def __init__(self, step=1, only_root=False):
        self.step = step
        self.only_root = only_root

    def on_run_heuristics(self, bb, A, b, node):
        if self.only_root:
            if node.get_level() == 0:
                primal_heur(bb, A, b, bb.get_ub(), node)
        elif bb.get_node_count() % self.step == 0:
            primal_heur(bb, A, b, bb.get_ub(), node)


class LagrPenaltiesReductionCallback(BBCallback):
    """Callback performing problem reduction using
    the Lagrangean penalties method.
    """
    def on_reduction(self, node, A, b, ub):
        lagr_penalties(node, A, ub)


class ColumnInclusionCallback(BBCallback):
    """Callback performing problem reduction using
    the column inclusion method.
    """
    def on_reduction(self, node, A, b, ub):
        column_inclusion(node, A, b)
