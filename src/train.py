"""All training functions are defined here.
"""
import torch

import environments
from models import BaseModel, LinearModel
from environments import MacroSokobanEnv
from search_tree import SearchTree, Node
from features import core_features
from variables import MAX_MICROSOKOBAN


def expand_node(
        tree: SearchTree,
        node: Node,
        model: BaseModel,
        gamma: bool
    ):
    """Expand the node and update all its children values.
    """
    node.expand(tree)  # Expand new nodes
    with torch.no_grad():
        for child in node.children:
            model.estimate(child, gamma=gamma)  # Evaluate child's value


def compute_loss(
        model: BaseModel,
        node: Node,
        gamma: bool
    ):
    """Compute the loss for the given node prediction.

    It predicts the node's value, and compare it with its children.
    The node must already have children with updated values!
    """
    # Find the best child and compute loss
    best_child = max(node.children, key=lambda node: node.value)
    prediction = model.estimate(node, gamma=gamma)  # Shape [1, 1]
    target = torch.FloatTensor([[gamma * best_child.value + best_child.reward]])  # Shape [1, 1]
    loss = (target - prediction).pow(2).sum()
    return loss


def train_on_env(model: BaseModel, env: MacroSokobanEnv, config: dict):
    """Example of a training loop on a single environment.
    """
    epsilon, seed, gamma = config['epsilon'], config['seed'], config['gamma']
    optimizer = config['optimizer']

    tree = SearchTree(env, epsilon, model, seed)
    model.estimate(tree.root, gamma)

    for leaf_node in tree.episode():
        expand_node(tree, leaf_node, model, gamma)

        if leaf_node.children:
            loss = compute_loss(model, leaf_node, gamma)

            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Backpropagate new model estimations
            tree.update_all_values(model)

    return tree.solution_path()


def train_backward(config: dict) -> LinearModel:
    """Train a basic linear model on all the backward tasks.
    It is trained on the MicroSokoban levels.
    One epoch is a passage through all the levels (155).

    Return the trained linear model.
    """
    env = MacroSokobanEnv(forward=False, dim_room=(6, 6), num_boxes=2)
    feat_size = len(core_features(env, config['gamma']))
    model = LinearModel(feat_size)
    optim = torch.optim.Adam(model.parameters(), lr=config['lr'])

    for _ in range(config['epochs']):
        for level_id in range(1, MAX_MICROSOKOBAN+1):
            env = environments.from_file(
                'MicroSokoban',
                level_id,
                forward=False,
                max_steps=config['max_steps'],
            )
            train_on_env(model, env, config)

    return model


if __name__ == '__main__':
    from models import LinearModel
    from utils import print_board, build_board_from_raw
    from features import core_features

    env = MacroSokobanEnv(forward=False, dim_room=(6, 6), num_boxes=2)
    feat = core_features(env, 1)
    model = LinearModel(len(feat))

    config = {
        'gamma': 0.9,
        'seed': 0,
        'epsilon': 0.1,
        'optimizer': torch.optim.Adam(model.parameters(), lr=1e-3),
    }

    solution = train_on_env(model, env, config)
    print('Solution')
    for node in solution:
        node.env.print()
        print('')
