"""Functions to compute the backward and forward features.
Those features are used to train the basic linear model.
"""
import numpy as np
from numba import njit

from environments import MacroSokobanEnv
from utils import connectivity, build_board_from_raw
from variables import TYPE_LOOKUP


def targets(env: MacroSokobanEnv) -> int:
    """Return the number of boxes already packed on
    target square.
    """
    return env.boxes_on_target / env.num_boxes


BOX_TARGET = TYPE_LOOKUP['box target']
BOX_NOT_ON_TARGET = TYPE_LOOKUP['box not on target']
@njit
def optimized_dist(board: np.array, num_boxes: int):
    if (board == BOX_TARGET).sum() == 0:
        return 0  # All box are placed

    total_distance = 0
    boxes_pos = np.argwhere(board == BOX_NOT_ON_TARGET)
    targets_pos = np.argwhere(board == BOX_TARGET)

    for box in boxes_pos:
        # Compute Manhattan distance with every empty target
        distance_from_each_target = np.sum(np.abs(box - targets_pos), axis=1)
        pos = np.argmin(distance_from_each_target)
        keep_indices = [i for i in range(len(targets_pos))]
        del keep_indices[pos]
        targets_pos = targets_pos[np.array(keep_indices)]
        total_distance += distance_from_each_target[pos]

    norm = num_boxes * (board.shape[0] + board.shape[1])
    return total_distance / norm


def distance(env: MacroSokobanEnv) -> int:
    """Return the total distance of the boxes from the targets.
    This is a lower bound of the number of moves required.
    """
    raw = env.render()
    board, _ = build_board_from_raw(raw)
    return optimized_dist(board, env.num_boxes)


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


#################
# Hint features #
#################
def find_box_pos(board: np.array) -> set[tuple]:
    """find all box positions and put their coordinates in a set.
    """
    positions = np.argwhere(
        (board == TYPE_LOOKUP['box on target']) |\
        (board == TYPE_LOOKUP['box not on target'])
    ).tolist()
    positions = set([tuple(p) for p in positions])
    return positions


def overlap(env: MacroSokobanEnv, solution: list) -> float:
    """Maximum number of boxes on the same position
    in the current board as any other board in the backward solution.
    """
    current_board = env.render()
    current_board, _ = build_board_from_raw(current_board)
    current_positions = find_box_pos(current_board)
    total_boxes = len(current_positions)

    box_positions = []
    for node in solution:
        board = node.env.render()
        board, _ = build_board_from_raw(board)
        box_positions.append(find_box_pos(board))

    max_overlap = 0
    for positions in box_positions:
        overlap = len(positions & current_positions)
        max_overlap = max(max_overlap, overlap)

    return max_overlap / total_boxes


def find_box_on_target(board: np.array) -> set[tuple]:
    """Return the set containing all the box positions
    that are on their target.
    """
    positions = np.argwhere(
        board == TYPE_LOOKUP['box on target']
    ).tolist()
    positions = set([tuple(p) for p in positions])
    return positions


def perm(env: MacroSokobanEnv, solution: list) -> float:
    """Count the number of boxes well placed on their target,
    according to the removing order found in the backward solution.
    """
    current_board = env.render()
    current_board, _ = build_board_from_raw(current_board)
    current_positions = find_box_on_target(current_board)
    total_boxes = len(find_box_pos(current_board))

    box_positions = []
    for node in solution[::-1]:
        board = node.env.render()
        board, _ = build_board_from_raw(board)
        box_positions.append(find_box_on_target(board))

    perm = 0
    while perm < len(box_positions) and current_positions == box_positions[perm]:
        perm += 1
    return perm / total_boxes


def all_features(env: MacroSokobanEnv, gamma: float, solution: list) -> list[float]:
    """Return all core features and the hint features for the forward agent.
    The solution is the backward solution found by the backward agent.
    """
    return core_features(env, gamma) + [overlap(env, solution), perm(env, solution)]
