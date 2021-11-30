"""Functions to compute the backward and forward features.
Those features are used to train the basic linear model.
"""
import numpy as np

from environments import MacroSokobanEnv
from utils import connectivity, build_board_from_raw
from variables import TYPE_LOOKUP


def targets(env: MacroSokobanEnv) -> int:
    """Return the number of boxes already packed on
    target square.
    """
    return env.boxes_on_target / env.num_boxes


def distance(env: MacroSokobanEnv) -> int:
    """Return the total distance of the boxes from the targets.
    This is a lower bound of the number of moves required.
    """
    raw = env.render()
    board, _ = build_board_from_raw(raw)

    if (board != TYPE_LOOKUP['box target']).all():
        return 0  # All box are placed

    total_distance = 0
    boxes_pos = np.argwhere(board == TYPE_LOOKUP['box not on target'])
    targets_pos = np.argwhere(board == TYPE_LOOKUP['box target']).tolist()

    for box in boxes_pos:
        distance_from_each_target = []
        for target in targets_pos:
            # Compute Manhattan distance with every empty target
            distance_from_each_target.append(np.sum(abs(box - target)))
        targets_pos.remove(targets_pos[np.argmin(distance_from_each_target)])
        total_distance += np.min(distance_from_each_target)

    norm = env.num_boxes * (env.dim_room[0] + env.dim_room[1])
    return total_distance / norm


def gamma1(env: MacroSokobanEnv, gamma: float) -> float:
    """A crude estimate of the reward for solving the level.
    It takes into account the discounting factor gamma.
    """
    return gamma**env.num_boxes


def gamma2(env: MacroSokobanEnv, gamma: float) -> int:
    """A refined version of gamma1, taking into account the boxes already on target.

    The feature depends whether the environment is in forward or backward mode.
    """
    if env.forward:
        return gamma**(env.num_boxes-env.boxes_on_target)
    else:
        return gamma**env.boxes_on_target


def core_features(env: MacroSokobanEnv, gamma: float) -> list[float]:
    """Return a list of all the core features.
    """
    return [
        targets(env),
        distance(env),
        gamma1(env, gamma),
        gamma2(env, gamma),
        len(connectivity(env)) / env.num_boxes,
    ]
