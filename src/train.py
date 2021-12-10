"""All training functions are defined here.
"""
import torch
import numpy as np

import environments
from models import BaseModel
from environments import MacroSokobanEnv
from search_tree import SearchTree, Node


def expand_node(
        tree: SearchTree,
        node: Node,
        model: BaseModel,
        gamma: bool,
        backward_sol: list = None
    ):
    """Expand the node and update all its children values.
    """
    node.expand(tree)  # Expand new nodes
    with torch.no_grad():
        for child in node.children:
            model.estimate(child, gamma=gamma, backward_solution=backward_sol)  # Evaluate child's value


def compute_loss(
        model: BaseModel,
        node: Node,
        gamma: bool,
        backward_sol: list = None,
        final_loss: bool = False,
    ):
    """Compute the loss for the given node prediction.

    It predicts the node's value, and compare it with its children.
    The node must already have children with updated values!
    """
    if final_loss:
        prediction = model.estimate(node, gamma=gamma, backward_solution=backward_sol)
        loss = (prediction).pow(2).sum()  # Target is 0
        return loss

    # Find the best child and compute loss
    best_child = max(node.children, key=lambda node: node.reward + gamma * node.value)
    prediction = model.estimate(node, gamma=gamma, backward_solution=backward_sol)  # Shape [1, 1]
    target = torch.FloatTensor([[gamma * best_child.value + best_child.reward]])  # Shape [1, 1]
    loss = (target - prediction).pow(2).sum()
    return loss


def eval_on_env(
        model: BaseModel,
        env: MacroSokobanEnv,
        config: dict,
        backward_sol: list = None,
    ):
    """Evaluate the model on the given environment.
    It does the same thing as the training loop, without training the model.
    """
    epsilon, seed, gamma = config['epsilon'], config['seed'], config['gamma']

    tree = SearchTree(env, epsilon, seed)
    with torch.no_grad():
        model.estimate(tree.root, gamma=gamma, backward_solution=backward_sol)

    total_loss = 0

    with torch.no_grad():
        for leaf_node in tree.episode():
            expand_node(tree, leaf_node, model, gamma, backward_sol)

            if leaf_node.children:
                loss = compute_loss(model, leaf_node, gamma, backward_sol)

                # Backpropagate new model estimations
                tree.update_all_values(model, gamma=gamma, backward_solution=backward_sol)

                total_loss += loss.cpu().item()

    return tree.solution_path(), total_loss / (tree.steps_count + 1)


def train_on_env(
        model: BaseModel,
        env: MacroSokobanEnv,
        config: dict,
        backward_sol: list = None
    ):
    """Training loop over one episode.
    It works for the backward and forward mode.
    """
    epsilon, seed, gamma = config['epsilon'], config['seed'], config['gamma']
    optimizer = config['optimizer']

    tree = SearchTree(env, epsilon, seed)
    with torch.no_grad():
        model.estimate(tree.root, gamma=gamma, backward_solution=backward_sol)

    total_loss = 0

    for leaf_node in tree.episode():
        if leaf_node.env._check_if_won():
            continue

        expand_node(tree, leaf_node, model, gamma, backward_sol)

        # Update when posible and when it is not the ending leaf
        if leaf_node.children and tree.steps_count < tree.max_steps:
            loss = compute_loss(model, leaf_node, gamma, backward_sol)

            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Backpropagate new model estimations
            tree.update_all_values(model, gamma=gamma, backward_solution=backward_sol)

            total_loss += loss.cpu().item()

    # At the end of an episode, we always have a reward of 0
    leaf_node.value = 0

    # Final loss update for the ending leaf
    loss = compute_loss(model, leaf_node, gamma, backward_sol, final_loss=True)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.cpu().item()

    return tree.solution_path(), total_loss / (tree.steps_count + 1)


def train_on_solution(
        model: BaseModel,
        solution: list[Node],
        config: dict,
        backward_sol: list[Node] = None,
    ):
    """Takes the solution of an episode and retrain
    the model on this solution (like a replay buffer of good quality).
    """
    gamma = config['gamma']
    optimizer = config['optimizer']
    total_loss = 0

    as_won = solution[-1].env._check_if_won()

    last_node_value = 0
    last_node_reward = 0
    for node in reversed(solution):
        target_value = last_node_reward + gamma * last_node_value

        # Compute loss
        prediction = model.estimate(node, gamma=gamma, backward_solution=backward_sol)  # Shape [1, 1]
        target = torch.FloatTensor([[target_value]])  # Shape [1, 1]
        loss = (target - prediction).pow(2).sum()

        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.cpu().item()

        last_node_value = target_value
        last_node_reward = node.reward

    return total_loss / len(solution)


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

    solution, _ = train_on_env(model, env, config)
    print('Solution')
    for node in solution:
        node.env.print()
        print('')
