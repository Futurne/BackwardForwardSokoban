"""A search tree for learning and inference.
It gives the most promising leaf to expand
and update each states values accordingly.

The leaf can be choosen in an epsilon-greedy fashion,
useful for the learning procedure.
"""
from copy import deepcopy

from numpy.random import default_rng
from numpy.random._generator import Generator

from environments import MacroSokobanEnv
from models import BaseModel


class Node:
    """Contains its copy of the environment and its estimated value.
    It's able to expand to new children.
    """
    def __init__(
            self,
            parent: Node,
            env: MacroSokobanEnv,
            reward: float,
            done: bool,
            ):
        """
        Args
        ----
            :parent:    The parent node of this node. None if this node is the root node.
            :env:       Environment of this node.
            :reward:    Reward obtained when reaching the state associated with this node.
            :done:      True if the state associated with this node is a final state.
        """
        self.parent = parent
        self.env = env
        self.reward = reward
        self.done = done

        self.children = []
        self.value = 0
        self.torch_value = 0

    def eval(self, model: BaseModel):
        """Update the value using the given :model: for the estimation.
        """
        self.torch_value = model.estimate(self.env.room_state)
        self.value = self.torch_value.float()

    def expand(self):
        """Expand all the possible children.
        Their values are NOT evaluated.
        """
        states = self.env.reachable_states()
        for room_state in states:
            env = deepcopy(self.env)
            obs, reward, done, info = env.step(room_state)
            self.children.append(Node(self, env, reward, done))

    def best_child(self, epsilon: float, rng: Generator):
        """Return the best child with a probability 1 - epsilon.
        Otherwise return a random child with a
        probability of epsilon.

        Can't be called if the node has no children.
        """
        assert len(self.children) > 0

        if rng.random() < epsilon:
            # Random node
            return rng.choice(self.children)
        return max(self.children, key=lambda node: node.value)



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
        self.root = Node(None, env, 0, False)
        self.root.eval(model)
        self.epsilon = epsilon
        self.rng = default_rng(seed)

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
        while not leaf.done:
            yield leaf
            leaf = self.next_leaf()
