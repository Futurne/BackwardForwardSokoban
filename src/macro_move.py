"""Utils function to generate macro moves.

A macro-move is a combination of legal moves of a unique box.
Theses macro-moves are equivalent to a multiple serie of movements,
but are much more efficient for the search tree.
"""
import numpy as np

from utils import find_neightbours
from variables import TYPE_LOOKUP


def macro_moves(board: np.array, player: np.array, box: np.array) -> list[np.array]:
    """Compute all states reachable by moving the given `box`
    by the player.
    Return the list of all states reachable.

    Each state is the result of making mutiple legal move in the board
    by the player when only moving or pushing the specified box.
    """
    available_cells = np.argwhere(
        (board == TYPE_LOOKUP['empty space']) |\
        (board == TYPE_LOOKUP['box target'])
    )
    available_cells = set((c[0], c[1]) for c in available_cells)
    neightbours = find_neightbours(tuple(player), available_cells)
    available_cells.add(tuple(box))  # For future computations

    visited = {board.data.tobytes()}
    to_visit = [(board.copy(), neightbours, box)]  # LIFO

    forbidden = {
        TYPE_LOOKUP['wall'],
        TYPE_LOOKUP['box on target'],
        TYPE_LOOKUP['box not on target'],
    }

    states = list()
    while to_visit:
        board, neightbours, box = to_visit.pop()
        x, y = box
        for c in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
            if c not in neightbours:
                continue

            opposite = box + (box - c)

            if board[tuple(opposite)] in forbidden:
                continue

            # Legal move
            # Perform movement
            board[tuple(opposite)] = TYPE_LOOKUP['box on target'] if board[tuple(opposite)] == TYPE_LOOKUP['box target']\
                    else TYPE_LOOKUP['box not on target']

            board[tuple(box)] = TYPE_LOOKUP['box target'] if board[tuple(box)] == TYPE_LOOKUP['box on target']\
                    else TYPE_LOOKUP['empty space']

            if board.data.tobytes() in visited:
                continue  # We already visited this board

            # Compute neightbours
            available = available_cells.copy()
            available.add(tuple(opposite))
            neightbours = find_neightbours(tuple(box), available)

            # Update lists and sets
            to_visit.append((board.copy(), neightbours, opposite))
            visited.add(board.data.tobytes())
            states.append(board.copy())

    return states


if __name__ == '__main__':
    from gym_sokoban.envs import sokoban_env
    from utils import build_board_from_raw

    env = sokoban_env.SokobanEnv((10, 10), num_boxes=2)
    raw = env.reset(render_mode='raw')
    board = build_board_from_raw(raw)

    player = np.argwhere(raw[-1] == 1)[0]
    box = np.argwhere(raw[2] == 1)[0]
    s = macro_moves(board, player, box)
    print(len(s))
