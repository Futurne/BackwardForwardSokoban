"""All training functions are defined here.
"""
import torch

from models import BaseModel
from environments import MacroSokobanEnv
from search_tree import SearchTree, Node


def expand_node(tree: SearchTree, node: Node, model: BaseModel):
    """Expand the node and update all its children values.
    """
    node.expand(tree)  # Expand new nodes
    with torch.no_grad():
        for child in node.children:
            model.estimate(child, gamma=0.9)  # Evaluate child's value


def compute_loss(model: BaseModel, node: Node):
    """Compute the loss for the given node prediction.

    It predicts the node's value, and compare it with its children.
    The node must already have children with updated values!
    """
    # Find the best child and compute loss
    best_child = max(node.children, key=lambda node: node.value)
    prediction = model.estimate(node, gamma=0.9)  # Shape [1, 1]
    target = torch.FloatTensor([[best_child.value + best_child.reward]])  # Shape [1, 1]
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
        expand_node(tree, leaf_node, model)

        if leaf_node.children:
            loss = compute_loss(model, leaf_node)

            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Backpropagate new model estimations
            tree.update_all_values(model)

    return tree.solution_path()


if __name__ == '__main__':
    from models import LinearModel
    from utils import core_feature, print_board, build_board_from_raw

    env = MacroSokobanEnv(forward=False, dim_room=(6, 6), num_boxes=2)
    feat = core_feature(env, 0.9)
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
