"""A search tree for learning and inference.
It gives the most promising leaf to expand
and update each states values accordingly.

The leaf can be choosen in an epsilon-greedy fashion,
useful for the learning procedure.
"""
import heapq  # priority queue
from copy import deepcopy

import torch
from numpy.random import default_rng
from numpy.random._generator import Generator

from environments import MacroSokobanEnv
from models import BaseModel
from utils import build_board_from_raw, is_board_deadlock
from variables import TYPE_LOOKUP


class Node:
    """Contains its copy of the environment and its estimated value.
    It's able to expand to new children.
    """
    def __init__(
            self,
            env: MacroSokobanEnv,
            parent: Node,
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
        self.is_deadlock = is_board_deadlock(env.room_state)
        self.children = []

    def expand(self, tree: SearchTree):
        """Expand all the possible children.
        Their values are NOT evaluated.
        """
        states = self.env.reachable_states()
        for room_state in states:
            env = deepcopy(self.env)
            obs, reward, done, info = env.step(room_state)

            raw = env.render()
            board, player = build_board_from_raw(raw)
            board[tuple(player)] = TYPE_LOOKUP['player']

            # Does this state has already been visited before?
            if board.data.tobytes() in tree.boards:
                # Do note create a new node
                continue

            # Create and add the new node to the boards and priority queue
            node = Node(env, self, reward, done)
            tree.boards.add(board.data.tobytes())
            heapq.heappush(tree.priority_queue, node)
            self.children.append(node)

        # Check for deadlocks
        for child in self.children:
            child.deadlock_removal()

    def best_child(self, model: BaseModel, epsilon: float, rng: Generator):
        """Return the best child with a probability 1 - epsilon.
        Otherwise return a random child with a
        probability of epsilon.

        Only the non-deadlock children are considered.

        Can't be called if the node has no children.
        """
        assert len(self.children) > 0

        if rng.random() < epsilon:
            # Random node
            return rng.choice(self.children)

        valid_childs = [child for child in self.children
                        if not child.is_deadlock]
        return max(valid_childs, key=lambda: child: child.value)

    def backprop(self, gamma: float=1):
        """Update this node's value based on its childs.
        """
        if self.children == []:
            return  # Nothing to backprop

        values = [child.reward + gamma * child.values
                  for child in self.children]
        self.value = max(values)

    def deadlock_removal(self, tree: SearchTree):
        """Declare this node as deadlock if it has only deadlock children.
        Remove this node from the priority queue if it is a deadlock.
        """
        if all([child.is_deadlock for child in self.children]):
            self.is_deadlock = True

        if self.is_deadlock and self in tree.priority_queue:
            tree.priority_queue.remove(self)

        self.parent.deadlock_removal(tree)

    def __lt__(self, other: Node):
        """Compare node's values.
        Useful for the priority queue.
        """
        return self.value < other.value


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
        self.root = Node(env, 0, False)
        self.root.eval(model)
        self.epsilon = epsilon
        self.rng = default_rng(seed)

        # A mapping between board_states and nodes of the tree
        self.boards = {env.room_state.data.tobytes()}
        self.priority_queue = [self.root]

    def next_leaf(self, model: BaseModel):
        """Start from the root node and follows the best
        child node until a leaf is reach.
        At each step, a random node can be choosen instead
        of the best node, with a probability of 1 - :self.epsilon:.
        """
        leaf = self.root
        while leaf.children:
            leaf = leaf.best_child(model, self.epsilon, self.rng)
        return leaf

    def episode(self, model: BaseModel):
        """Play an episode. Yield the current leaf each time so
        an exterior model can expand them.
        The episode stops once a ending leaf is reach (leaf.done is True).
        """
        leaf = self.next_leaf(model)
        while not leaf.done:
            yield leaf
            leaf = self.next_leaf(model)

    def leafs(self):
        """Return all leafs of the tree.
        """
        leafs = [node for node in self.priority_queue
                 if node.children == []]
        return leafs

    def backpropagate_values(self):
        """Backpropagate all the leaf values up to the root node.
        Make sure that every leafs of the tree has an updated value.
        """
        # Need to backpropagate the deepest nodes first!
        for node in self.priority_queue[::-1]:
            node.backprop()


"""
for leaf in tree.episode(model):
    env = leaf.env
    value = model.predict(env)
    leaf.expand()
    max_value = -inf
    reward = 0

    for child in leaf.children:
        value_child = model.predic(child.env)
        if value_child > max_value:
            max_value = value_child
            reward = child.reward

    optimizer.zero_grad()
    loss = (reward + gamma * max_value - value).pow(2)
    loss.backward()
    optimizer.step()
"""
