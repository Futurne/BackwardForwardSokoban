"""All training functions are defined here.
"""
from models import BaseModel
from environments import MacroSokobanEnv
from search_tree import SearchTree


def train_on_env(model: BaseModel, env: MacroSokobanEnv):
    """Example of a training loop on a single environment.
    """
    epsilon = 0.1
    seed = 0
    gamma = 0.99
    tree = SearchTree(env, epsilon, model, seed)
    for leaf_node in tree.episode():
        leaf_node.expand()  # Expand new nodes
        for child in leaf_node.children:
            child.eval(model)  # Evaluate child's value

        # Find the best child and compute target
        best_child = max(leaf_node.childre, key=lambda node: node.value)
        target = best_child.torch_reward + best_child.torch_value
        # Compute loss
        loss = (target.detach() - leaf_node.torch_value).pow(2)
        # Update the new value estimate of the node and all its parents
        leaf_node.update_value(target)
