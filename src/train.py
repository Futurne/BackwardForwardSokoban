"""All training functions are defined here.
"""
import torch

from models import BaseModel
from environments import MacroSokobanEnv
from search_tree import SearchTree, Node


def forward_train(model: BaseModel, env: MacroSokobanEnv, config: dict):
    """Example of a training loop on a single environment.
    """
    epsilon, seed, gamma = config['epsilon'], config['seed'], config['gamma']
    optimizer = config['optimizer']

    tree = SearchTree(env, epsilon, model, seed)
    model.estimate(tree.root, gamma)

    print('Initial board:')
    raw = tree.root.env.render()
    board, player = build_board_from_raw(raw)
    print_board(board, player)

    for leaf_node in tree.episode():
        print('\nExpending leaf:')
        raw = leaf_node.env.render()
        board, player = build_board_from_raw(raw)
        print_board(board, player)

        leaf_node.expand(tree)  # Expand new nodes
        with torch.no_grad():
            for child in leaf_node.children:
                model.estimate(child, gamma)  # Evaluate child's value

        if leaf_node.children:
            # Find the best child and compute loss
            best_child = max(leaf_node.children, key=lambda node: node.value)
            prediction = model.estimate(leaf_node, gamma)  # Shape [1, 1]
            target = torch.FloatTensor([[best_child.value + best_child.reward]])  # Shape [1, 1]
            loss = (target - prediction).pow(2).sum()

            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Backpropagate new model estimations
            tree.update_all_values(model)


if __name__ == '__main__':
    from models import LinearModel
    from utils import core_feature, print_board, build_board_from_raw

    env = MacroSokobanEnv(dim_room=(10, 10), num_boxes=3)
    feat = core_feature(env, 0.9)
    model = LinearModel(len(feat))

    config = {
        'gamma': 0.9,
        'seed': 0,
        'epsilon': 0.1,
        'optimizer': torch.optim.Adam(model.parameters(), lr=1e-3),
    }

    forward_train(model, env, config)
