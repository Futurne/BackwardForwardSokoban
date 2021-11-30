"""A search tree for learning and inference.
It gives the most promising leaf to expand
and update each states values accordingly.

The leaf can be choosen in an epsilon-greedy fashion,
useful for the learning procedure.
"""
import heapq  # priority queue
from copy import deepcopy

import torch
import numpy as np
from numpy.random import default_rng
from numpy.random._generator import Generator

from environments import MacroSokobanEnv
from models import BaseModel
from utils import build_board_from_raw, is_env_deadlock
from variables import TYPE_LOOKUP


class Node:
    """Contains its copy of the environment and its estimated value.
    It's able to expand to new children.
    """
    def __init__(
            self,
            env: MacroSokobanEnv,
            parent,
            reward: float,
            done: bool,
            ):
        """
        Args
        ----
            :env:       Environment of this node.
            :parent:    The parent node of this node. None if this node is the root node.
            :reward:    Reward obtained when reaching the state associated with this node.
            :done:      True if the state associated with this node is a final state.
        """
        self.env = env
        self.parent = parent
        self.reward = reward
        self.done = done

        self.depth = 1 + parent.depth if parent else 0  # How deep is the node in the tree
        self.value = None
        self.children = []

        self.is_deadlock = is_env_deadlock(env)

    def expand(self, tree):
        """Expand all the possible children.
        Their values are NOT evaluated.
        """
        for room_state in self.env.reachable_states():
            env = deepcopy(self.env)
            obs, reward, done, info = env.step(room_state)
            # Does this state has already been visited before?
            if tree.is_visited_already(env):
                # Do note create a new node
                continue

            # Create and add the new node to the boards and priority queue
            node = Node(env, self, reward, done)
            tree.add_to_visited(env)
            heapq.heappush(tree.priority_queue, node)
            self.children.append(node)

        # Check for deadlocks
        if self.children:
            for child in self.children:
                child.deadlock_removal(tree)
        else:
            # This node couldn't expand to childs => he's useless
            # This can happen if all its possible childs are already in the `tree.boards`
            self.is_deadlock = True
            self.deadlock_removal(tree)

    def best_child(self, epsilon: float, rng: Generator):
        """Return the best child with a probability 1 - epsilon.
        Otherwise return a random child with a
        probability of epsilon.

        Only the non-deadlock children are considered.

        Can't be called if the node has no children.
        """
        assert len(self.children) > 0

        valid_childs = [child for child in self.children
                        if not child.is_deadlock]

        if rng.random() < epsilon:
            # Random node
            return rng.choice(valid_childs)

        return max(valid_childs, key=lambda child: child.value)

    def backprop(self, gamma: float=1):
        """Update this node's value based on its childs.
        """
        if self.children == []:
            return  # Nothing to backprop

        values = [child.reward + gamma * child.value
                  for child in self.children]
        self.value = max(values)

    def deadlock_removal(self, tree):
        """Declare this node as deadlock if it has only deadlock children.
        Remove this node from the priority queue if it is a deadlock.
        """
        # Update to deadlock state if all childs are in deadlock state
        if self.children and all([child.is_deadlock for child in self.children]):
            self.is_deadlock = True

        # Remove the node from the priority queue if necessary
        if self.is_deadlock and self in tree.priority_queue:
            tree.priority_queue.remove(self)

        # If in a deadlock state, try updating the parent node
        if self.is_deadlock and self.parent:
            self.parent.deadlock_removal(tree)

    def __lt__(self, other):
        """Compare node's depth.
        Useful for the priority queue.
        """
        return self.depth < other.depth


class SearchTree:
    """A searching tree, able to give the current most promising
    leaf. It also automatically expand and evaluates leafs.
    It's an interface between the agent and the environment.

    Can be used as an inference or learning tool for the Sokoban.
    """
    def __init__(
            self,
            env: MacroSokobanEnv,
            epsilon: float,
            model: BaseModel,
            seed: int,
            ):
        self.root = Node(env, None, 0, False)
        self.epsilon = epsilon
        self.rng = default_rng(seed)
        self.visited = set()
        self.last_leaf = None  # Final leaf of the episode

        self.add_to_visited(self.root.env)
        raw = np.array(env.render())
        self.priority_queue = [self.root]

    def next_leaf(self):
        """Start from the root node and follows the best
        child node until a leaf is reach.
        At each step, a random node can be choosen instead
        of the best node, with a probability of 1 - :self.epsilon:.
        """
        leaf = self.root
        while leaf.children:
            leaf = leaf.best_child(self.epsilon, self.rng)
        return leaf

    def episode(self):
        """Play an episode. Yield the current leaf each time so
        an exterior model can expand them.
        The episode stops once a ending leaf is reach (leaf.done is True).
        """
        leaf = self.next_leaf()
        yield leaf
        while not leaf.done:
            leaf = self.next_leaf()
            yield leaf

        self.last_leaf = leaf

    def solution_path(self):
        """Return the solution path from the root node
        to the last leaf used by the last episode.
        """
        assert self.last_leaf, "No episode has been done"

        current_node = self.last_leaf
        nodes = [current_node]
        while current_node.parent:
            current_node = current_node.parent
            nodes = [current_node] + nodes

        return nodes

    def leafs(self):
        """Return all leafs of the tree.
        """
        leafs = [node for node in self.priority_queue
                 if node.children == []]
        return leafs

    def update_all_values(self, model: BaseModel):
        """Backpropagate all the leaf values up to the root node.
        It makes sure that every leafs of the tree has an updated value.
        """
        for leaf_node in self.leafs():
            model.estimate(leaf_node, gamma=0.9)  # Eval leaf's value

        # Need to backpropagate the deepest nodes first!
        for node in self.priority_queue[::-1]:
            node.backprop()

    @staticmethod
    def _board_to_str(board: np.array) -> str:
        """Util function to create a string
        based on the board. Each cell value is a word.
        """
        board = [
            str(cell)
            for row in board
            for cell in row
        ]
        board = ' '.join(board)  # str representation
        return board

    def add_to_visited(self, env: MacroSokobanEnv):
        """Add the node to the visited nodes' set.

        The hash function is turning the board into a long
        string of each cells.
        """
        board = SearchTree._board_to_str(env.room_state)
        self.visited.add(board)

    def is_visited_already(self, env: MacroSokobanEnv) -> bool:
        """Return whether this node has already been visited
        or not.

        A node is visited if it has been noticed to the tree using
        the `SearchTree.add_to_visited` method.
        """
        board = SearchTree._board_to_str(env.room_state)
        return board in self.visited
