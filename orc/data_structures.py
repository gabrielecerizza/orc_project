import time
from heapq import heappop, heappush

import numpy as np


class Node:
    """Node of a tree in a branch-and-bound data structure.

    Parameters
    ----------
    id : int
        A unique identifier for the node.

    branch_strategy : func
        Function called when generating children.

    lb_strategy : func
        Function called to compute the lower bound.

    level : int
        Level of the node in the tree.

    x0 : list of int
        Indices of variables fixed to 0.

    x1 : list of int
        Indices of variables fixed to 1.
    """
    def __init__(self, id, branch_strategy, lb_strategy, 
                 level=0, x0=None, x1=None):
        
        # Id used to sort nodes inside the heap
        # when two nodes have the same lb.
        self.id = id
        
        self.level = level
        self.x0 = [] if x0 is None else x0
        self.x1 = [] if x1 is None else x1
        self.branch_strategy = branch_strategy
        self.lb_strategy = lb_strategy
        self.lb = None
        self.x_lb = None
        self.lambd = None
        self.val = None

    def generate_children(self, A, b, bb):
        return self.branch_strategy(A, b, bb, self)

    def is_leaf(self, A, b):
        """Return true if the node can be considered 
        a leaf.
        
        If the partial solution is feasible, then we 
        can fix to 0 all the unassigned variables to 
        obtain the minimum of the subtree.
        """
        
        return self.is_feasible(A, b)
    
    def is_feasible(self, A, b):
        """Return true if the variables fixed to 1 are 
        enough to cover all the rows.
        """
        
        x = np.zeros((A.shape[-1],))
        x[self.x1] = 1
        return np.all(A @ x >= b)
    
    def is_infeasible(self, A, b):
        """Return true if we cannot cover all the rows even
        if we fixed to 1 all the unassigned variables.
        """
        
        x = np.ones((A.shape[-1],))
        x[self.x0] = 0
        return np.any(A @ x < b)
    
    def compute_lb(self, A, b, ub):
        self.lb = \
            self.lb_strategy(A, b, ub, 
                             self.x0[:], self.x1[:], self)
        
    def add_to_x0(self, ls):
        self.x0 = list(set(self.x0).union(set(ls)))

    def add_to_x1(self, ls):
        self.x1 = list(set(self.x1).union(set(ls)))
        
    def get_id(self):
        return self.id
    
    def get_level(self):
        return self.level
    
    def get_val(self, A):
        if self.val is None:
            x = np.zeros((A.shape[-1],))
            x[self.x1] = 1
            self.val = np.sum(A, axis=0) @ x
        return self.val
    
    def get_lb(self):
        return self.lb
    
    def get_x_lb(self):
        return self.x_lb
    
    def get_lambd(self):
        # Return a copy to avoid modifications
        return self.lambd[:]
    
    def get_x0(self):
        # Return a copy to avoid modifications
        return self.x0[:]
    
    def get_x1(self):
        # Return a copy to avoid modifications
        return self.x1[:]
    
    def get_num_var(self):
        return self.num_var
    
    def get_lb_strategy(self):
        return self.lb_strategy
    
    def set_x_lb(self, x_lb):
        self.x_lb = x_lb

    def set_lambd(self, lambd):
        self.lambd = lambd
    
    def __str__(self):
        return f"Node(level={self.level}, id={self.id}, " + \
            f"x0={self.x0}, x1={self.x1}, " + \
            f"val={self.val}, " + \
            f"lb={self.lb}, x_lb={self.x_lb}, " + \
            f"lambd={self.lambd})"
    

class Tree:
    """Tree containing the nodes used in a 
    branch-and-bound data structure.

    Parameters
    ----------
    root : Node
        The root of the tree.
    """
    def __init__(self, root):
        # Nodes are added to the open list as tuples
        # (node lower bound, id, node). The list is a heap
        # that returns the node with the lowest lower bound
        # upon calling pop.
        self.open_list = [(-np.inf, 0, root)]

    def is_empty(self):
        return len(self.open_list) == 0
    
    def remove(self):
        return heappop(self.open_list)[2]
    
    def add(self, node):
        heappush(self.open_list, 
                 (node.get_lb(), node.get_id(), node))


class TimeLimitException(Exception):
    """Exception raised when the runtime of the 
    branch-and-bound algorithm has exceeded a given
    threshold.
    """
    pass


class BranchAndBound:
    """Branch-and-bound data structure to solve a MIP problem.

    Parameters
    ----------
    branch_strategy : func
        Function called when generating children from a node.

    lb_strategy : func
        Function called to compute the lower bound.

    callbacks : list of func
        List of functions with callback methods called
        during the execution of the algorithm.

    time_start : float
        Starting time of the execution of the algorithm.

    time_limit : int
        Maximum number of seconds allowed for the
        execution of the algorithm.

    verbose : int
        Level of logging. When 1, logs are printed. When 2,
        logs are written on a file.
    """
    def __init__(
            self, branch_strategy, lb_strategy, 
            callbacks=None, time_start=None, 
            time_limit=60 * 5, verbose=0):
        self.tree = Tree(
            Node(0, branch_strategy, lb_strategy))
        self.node_count = 1
        self.best = None
        self.ub = np.inf
        self.x_best = None
        self.callbacks = callbacks or []
        self.time_start = time_start
        self.time_limit = time_limit
        self.verbose = verbose

    def search(self, A, b):
        while (not self.tree.is_empty()):
            if self.time_start is not None:
                elapsed = time.process_time() - self.time_start
                if elapsed > self.time_limit:
                    raise TimeLimitException
            node = self.tree.remove()
            self.log("search", "removed", str(node))
            self.run_heuristics(np.copy(A), np.copy(b), node)
            self.log("search", "ran heuristics", str(node))
            if node.get_level() == 0:
                node.compute_lb(np.copy(A), np.copy(b), self.ub)
                self.log("search", "root lb", str(node))
            self.branch(node, A, b)
            
    def run_heuristics(self, A, b, node):
        """Run heuristics to find new upper bounds 
        before branching.
        """
        for callback in self.callbacks:
            callback.on_run_heuristics(self, A, b, node)

    def reduction(self, node, A, b):
        """Attempt to reduce the size of the problem and
        possibly close the node if the node is revealed
        to be a leaf or infeasible after reduction.
        """

        # Keep looping as long as a callback has reduced
        # the problem, to see if other callbacks can further
        # reduce the problem.
        while (True):
            reduced = False
            for callback in self.callbacks:
                reduced = reduced or callback.on_reduction(
                    node, A, b, self.ub)
                if node.is_leaf(A, b):
                    self.evaluate_leaf(node, A)
                    self.log("reduction", "leaf", str(node))
                    return True
                elif node.is_infeasible(A, b):
                    self.log("reduction", "infeasible", str(node))
                    return True
            if (not reduced):
                break
        return False

    def branch(self, node, A, b):
        self.log("branch", "entered", str(node))
        closed = self.reduction(node, np.copy(A), np.copy(b))
        if closed:
            self.log("branch", "closed", str(node))
            return
        for child in node.generate_children(np.copy(A), 
                                            np.copy(b), self):
            self.log("branch", "generated child", str(child))
            if child.is_leaf(A, b):
                self.log("branch", "child is leaf", str(child))
                self.evaluate_leaf(child, A)
            elif not child.is_infeasible(A, b):
                child.compute_lb(np.copy(A), np.copy(b), self.ub)
                self.log("branch", "child lb", str(child))
                if child.get_lb() < self.ub:
                    self.log("branch", 
                             "child added to list", str(child)) 
                    self.tree.add(child)
            else:
                self.log("branch", 
                         "child infeasible", str(child)) 
                       
    def evaluate_leaf(self, leaf, A):
        """When the node is a leaf, compute the value
        of the solution and update the upper bound if
        it is better than the previous incumbent upper
        bound.
        """
        leaf_val = leaf.get_val(A)
        if (self.best is None) or \
            (leaf_val < self.ub):
            self.best = leaf
            self.ub = leaf_val
            x = np.zeros((A.shape[-1],))
            x[leaf.get_x1()] = 1
            self.x_best = x

    def log(self, method, event, *args):
        l = f"[{method}]: {event}, {args}\n"
        if self.verbose == 1:
            print(l)
        elif self.verbose == 2:
            with open("logs/log.txt", "a") as f:
                f.write(l)

    def get_new_id(self):
        id = self.node_count
        self.node_count += 1
        return id
    
    def get_ub(self):
        return self.ub

    def get_node_count(self):
        return self.node_count

    def set_ub(self, ub):
        self.ub = ub

    def set_x_best(self, x):
        self.x_best = x

    def set_best(self, node):
        self.best = node
