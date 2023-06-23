from heapq import heappop, heappush

import numpy as np


class Node:
    def __init__(self, id, branch_strategy, 
                 lb_strategy, level=0, 
                 x0=None, x1=None):
        
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
        
        # If the partial solution is feasible,
        # then we can fix to 0 all the unassigned
        # variables to obtain the minimum of the
        # subtree.
        return self.is_feasible(A, b)
    
    def is_feasible(self, A, b):
        
        # True if the variables fixed to 1 are
        # enough to cover all the rows. 
        x = np.zeros((A.shape[-1],))
        x[self.x1] = 1
        return np.all(A @ x >= b)
    
    def is_infeasible(self, A, b):
        
        # True if we cannot cover all the rows even
        # if we fixed to 1 all the unassigned variables.
        x = np.ones((A.shape[-1],))
        x[self.x0] = 0
        return np.any(A @ x < b)
    
    def compute_lb(self, A, b, ub):
        self.lb = \
            self.lb_strategy(A, b, ub, self.x0, self.x1, self)
        
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
        return self.lambd
    
    def get_x0(self):
        return self.x0
    
    def get_x1(self):
        return self.x1
    
    def get_num_var(self):
        return self.num_var
    
    def get_lb_strategy(self):
        return self.lb_strategy
    
    def set_x_lb(self, x_lb):
        self.x_lb = x_lb

    def set_lambd(self, lambd):
        self.lambd = lambd
    
    def __str__(self):
        return f"Node(level={self.level}, " + \
            f"x0={self.x0}, x1={self.x1}, " + \
            f"val={self.val}, " + \
            f"lb={self.lb}, x_lb={self.x_lb}, " + \
            f"lambd={self.lambd})"
    

class Tree:
    def __init__(self, root):
        self.open_list = [(-np.inf, 0, root)]

    def is_empty(self):
        return len(self.open_list) == 0
    
    def remove(self):
        return heappop(self.open_list)[2]
    
    def add(self, node):
        heappush(self.open_list, 
                 (node.get_lb(), node.get_id(), node))


class BranchAndBound:
    def __init__(
            self, branch_strategy, lb_strategy, 
            callbacks=None):
        self.tree = Tree(
            Node(0, branch_strategy, lb_strategy))
        self.node_count = 1
        self.best = None
        self.ub = np.inf
        self.x_best = None
        self.callbacks = callbacks or []

    def search(self, A, b):
        self.preprocess(A, b)
        while (not self.tree.is_empty()):
            node = self.tree.remove()
            if node.get_level() == 0:
                node.compute_lb(A, b, self.ub)
            self.branch(node, A, b)
            
    def preprocess(self, A, b):
        for callback in self.callbacks:
            callback.on_preprocess(self, A, b)

    def reduction(self, node, A, b):
        for callback in self.callbacks:
            callback.on_reduction(node, A, self.ub)
            if node.is_leaf(A, b):
                self.evaluate_leaf(node, A)
                return True
            elif node.is_infeasible(A, b):
                return True
        return False

    def branch(self, node, A, b):
        closed = self.reduction(node, A, b)
        if closed: return
        for child in node.generate_children(A, b, self):
            if child.is_leaf(A, b):
                self.evaluate_leaf(child, A)
            elif not child.is_infeasible(A, b):
                child.compute_lb(A, b, self.ub)
                if child.get_lb() < self.ub:   
                    self.tree.add(child)
                
            
    def evaluate_leaf(self, leaf, A):
        leaf_val = leaf.get_val(A)
        if (self.best is None) or \
            (leaf_val < self.best.get_val(A)):
            self.best = leaf
            self.ub = leaf_val
            x = np.zeros((A.shape[-1],))
            x[leaf.get_x1()] = 1
            self.x_best = x

    def get_new_id(self):
        id = self.node_count
        self.node_count += 1
        return id

    def set_ub(self, ub):
        self.ub = ub

    def set_x_best(self, x):
        self.x_best = x

    def set_best(self, node):
        self.best = node


def branch_strategy(A, b, bb, node):
    x0 = node.get_x0()
    x1 = node.get_x1()
    x_partial = np.zeros((A.shape[-1],))
    x_partial[node.get_x1()] = 1
    rc = (1 - node.get_lambd()) @ A

    # print("A @ x_partial", A @ x_partial)
    # print("b", b)
    # print("x_partial", x_partial)
    
    # Pick the row with the largest violation given the
    # solution obtained by completing the partial solution
    # of the node by fixing all the remaining variables
    # to 0.
    violation = b - (A @ x_partial)
    r = np.argmax(violation)
    
    # Columns with fixed values or with a zero entry
    # on the picked row are not candidates for branching.
    zero_entry = set(np.where(A[r] == 0)[0])
    not_candidates = list(set(x0).union(set(x1)).union(zero_entry))
    
    # Pick the column with minimum reduced cost
    rc[not_candidates] = np.inf
    j = np.argmin(rc)
    assert not j in not_candidates, (r, rc, x0, x1, A, b)

    return [
        Node(id=bb.get_new_id(),
             level=node.get_level() + 1, 
             x0=x0 + [j], 
             x1=x1, 
             branch_strategy=branch_strategy, 
             lb_strategy=node.get_lb_strategy()),
        Node(id=bb.get_new_id(),
             level=node.get_level() + 1, 
             x0=x0, 
             x1=x1 + [j], 
             branch_strategy=branch_strategy, 
             lb_strategy=node.get_lb_strategy())    
    ]


def branch_strategy2(A, b, bb, node):
    x0 = node.get_x0()
    x1 = node.get_x1()
    x_partial = np.zeros((A.shape[-1],))
    x_partial[node.get_x1()] = 1
    c = np.sum(A, axis=0)

    # print("A @ x_partial", A @ x_partial)
    # print("b", b)
    # print("x_partial", x_partial)
    
    # Pick the row with the largest violation given the
    # solution obtained by completing the partial solution
    # of the node by fixing all the remaining variables
    # to 0.
    violation = b - (A @ x_partial)
    r = np.argmax(violation)
    
    # Columns with fixed values or with a zero entry
    # on the picked row are not candidates for branching.
    zero_entry = set(np.where(A[r] == 0)[0])
    not_candidates = list(set(x0).union(set(x1)).union(zero_entry))
    
    # Pick the column with minimum reduced cost
    c[not_candidates] = np.inf
    j = np.argmin(c)
    assert not j in not_candidates, (r, c, x0, x1, A, b)

    return [
        Node(id=bb.get_new_id(),
             level=node.get_level() + 1, 
             x0=x0 + [j], 
             x1=x1, 
             branch_strategy=branch_strategy2, 
             lb_strategy=node.get_lb_strategy()),
        Node(id=bb.get_new_id(),
             level=node.get_level() + 1, 
             x0=x0, 
             x1=x1 + [j], 
             branch_strategy=branch_strategy2, 
             lb_strategy=node.get_lb_strategy())    
    ]
