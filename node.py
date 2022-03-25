import numpy as np
from fourier import Fourier
from functools import cache
"""
The Node class is for building decision trees recursively
Each instance of this class is node (either leaf of non-leaf)
"""


class Node:
    leaf_no = 0
    depth = 0
    no_leaves = 0
    leaf_no_to_node = {}

    def __init__(self, value=None, label=None, left=None, right=None):
        # self.value is the real number in the root if the root is a leaf and None otherwise
        self.value = value
        # self.label is the number of the variable assigned to the root (can be None if the root is a leaf)
        self.label = label
        # self.left is the left subtree (can be None)
        self.left = left
        # self.right is the right subtree (can be None)
        self.right = right

    def __str__(self):
        return "label:" + str(self.label) + " value:" +  str(self.value) +  " left:" +  str(self.left) +  "right:" + str(self.right)

    """
    # Returns a Fourier series (instance of the Fourier class) that defines the same function as the decision tree 
    defined by this object
    """
    def get_fourier(self):
        if self.is_leaf():
            return Fourier({frozenset(): self.value})

        else: # it is a variable node
            label = self.label
            assert label < n_var, "n_var is too small - the tree uses variables with higher numbers"
            fourier_left = self.left.get_fourier(n_var)
            fourier_right = self.right.get_fourier(n_var)
            new_series = {}

            for key in fourier_left.series:
                new_series[key] = new_series.get(key, 0) + (fourier_left.series[key] / 2)
                new_series[key.union([label])] = new_series.get(key.union([label]), 0) + (fourier_left.series[key] / 2)

            for key in fourier_right.series:
                new_series[key] = new_series.get(key, 0) + (fourier_right.series[key] / 2)
                new_series[key.union([label])] = new_series.get(key.union([label]), 0) - (fourier_right.series[key] / 2)

            return Fourier(new_series)

    """
    Recursively compute depth of the tree
    """
    @cache
    def tree_depth(self):
        if self.is_leaf():
            return 0
        else:
            return max(self.left.tree_depth(), self.right.tree_depth()) + 1
    """
    Recursively compute node count of the tree
    """
    @cache
    def node_count(self):
        if self.is_leaf():
            return 1
        else:
            return 1 + self.left.node_count() + self.right.node_count()

    """
    Recursviely compute number of leaves in the tree
    """
    @cache
    def leave_count(self):
        if self.is_leaf():
            return 1
        else:
            return self.left.leave_count() + self.right.leave_count()

    """
    Get tree prediction for input x
    """
    @cache
    def predict(self, X):
        predictions = []
        for row in X:
            predictions.append(self.__getitem__(row))
        return np.array(predictions)

    """
    Returns true if node is a leaf
    """
    def is_leaf(self):
        return self.value != None

    """
    If that coordinate has value 0 we go left and if it has value 1 we go right. 
    The value of the function is the real number in the leaf we reach.
    """
    def __getitem__(self, argument):
        if self.is_leaf():
            return self.value
        else:
            if argument[self.label] == 0:
                return self.left[argument]
            elif argument[self.label] == 1:
                return self.right[argument]
            else:
                raise Exception("argument can only contain 0s and 1s")

    @staticmethod
    def build_tree_from_sklearn(sklearn_tree):





        return

if __name__ == "__main__":
    #Leaf nodes
    A = Node(value=10)
    B = Node(value=20)
    C = Node (value=30)
    D = Node(value=40)
    # Variable nodes
    x1 = Node(label=1, left=A, right=B )
    x2 = Node(label=2, left=C, right=D)
    x0 = Node(label=0, left=x1, right=x2)
