"""Bunch of utils functions for the game of Sokoban.
"""
import numpy as np

from variables import TYPE_LOOKUP


def print_board(board: np.array):
    """Print the board in a human-readable way.
    """
    print(*board, sep='\n')


def build_board_from_raw(raw_board: np.array) -> np.array:
    """Merge the differents layers of the raw board into a single 2D board.

    The number corresponding of each type of cell is given by `variables.TYPE_LOOKUP`.
    """
    walls, goals, boxes, player = raw_board
    board = ((walls | goals | boxes) == 0) * TYPE_LOOKUP['empty space']
    board += (walls == 1) * TYPE_LOOKUP['wall']
    # Do we need the player ? I think not
    # board += player[player == 1] * TYPE_LOOKUP['player']
    board += (goals & boxes) * TYPE_LOOKUP['box on target']  # Find boxes and target
    board += ((goals ^ boxes) & goals) * TYPE_LOOKUP['box target']  # Find target without boxes
    board += ((boxes ^ goals) & boxes) * TYPE_LOOKUP['box not on target']  # Find boxes without target
    return board


def find_neightbours(cell: tuple, available_cells: set[tuple]) -> list[tuple]:
    """Find all neightbours in the `available_cells`.

    Return those neightbours and remove them from the `available_cells` set.
    """
    neightbours = [cell]
    to_visit = [cell]  # LIFO

    while to_visit:
        x, y = to_visit.pop()
        for n in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
            if n in available_cells:
                neightbours.append(n)
                to_visit.append(n)
                available_cells.remove(n)
    return neightbours


def connectivity(board: np.array) -> list[set]:
    """Find all group of cells that are freely walkable
    by the player.

    Return a list of set of cells. Each set contains
    the cells that can be reach by each other.
    """
    available_cells = np.argwhere(
        (board == TYPE_LOOKUP['empty space']) |\
        (board == TYPE_LOOKUP['box target'])
    )
    # Transforms the numpy array into a set of tuples
    available_cells = set((c[0], c[1]) for c in available_cells)
    neightbours = []

    while available_cells:
        current_cell = available_cells.pop()
        n = find_neightbours(current_cell, available_cells)
        neightbours.append(set(n))

    return neightbours


if __name__ == '__main__':
    from gym_sokoban.envs import sokoban_env
    env = sokoban_env.SokobanEnv((10, 10), num_boxes=2)
    raw = env.reset(render_mode='raw')
    board = build_board_from_raw(raw)
    print_board(board)
    n = connectivity(board)
    print(len(n))
