"""Utils function to generate macro moves.

A macro-move is a combination of legal moves of a unique box.
Theses macro-moves are equivalent to a multiple serie of movements,
but are much more efficient for the search tree.
"""
import numpy as np
from numba import jit

from utils import find_neighbours, yield_neighbours, print_board
from variables import TYPE_LOOKUP


FORBIDDEN = {
    TYPE_LOOKUP['wall'],
    TYPE_LOOKUP['box on target'],
    TYPE_LOOKUP['box not on target'],
}


def check_push(
    board: np.array,
    reachable_cells: list[tuple],
    player: (int, int),
    box: np.array,
    ) -> bool:
    """Check if the player can push the box.
    The player is supposed to be next to the box.
    """
    if player not in reachable_cells:  # Check if the box is reachable
        return False

    opposite = box + (box - player)
    if board[tuple(opposite)] in FORBIDDEN:  # Check if the move is legal
        return False

    return True


def check_pull(
    board: np.array,
    reachable_cells: list[tuple],
    player: (int, int),
    box: np.array,
    ) -> bool:
    """Check if the player can pull the box.
    The player is supposed to be next to the box.
    """
    if player not in reachable_cells:  # Check if the box is reachable
        return False

    behind = np.array(player) - (box - player)
    if board[tuple(behind)] in FORBIDDEN:  # Check if the move is legal
        return False

    return True


def apply_push(
    board: np.array,
    player: (int, int),
    box: np.array,
    ) -> np.array:
    """Push the box by the player next to it.
    Returns the new board. Doesn't modify the given board.

    After this function, we can consider that the new
    box and player positions are :opposite: and :box:
    respectively.
    """
    opposite = box + (box - player)
    board = board.copy()

    board[tuple(opposite)] = TYPE_LOOKUP['box on target'] if board[tuple(opposite)] == TYPE_LOOKUP['box target']\
            else TYPE_LOOKUP['box not on target']

    board[tuple(box)] = TYPE_LOOKUP['box target'] if board[tuple(box)] == TYPE_LOOKUP['box on target']\
            else TYPE_LOOKUP['empty space']

    return board


def apply_pull(
    board: np.array,
    player: (int, int),
    box: np.array,
    ) -> np.array:
    """Pull the box by the player next to it.
    Returns the new board. Doesn't modify the given board.

    After this function, we can consider that the new
    box and player positions are :player: and :behind:
    respectively.
    """
    behind = np.array(player) - (box - player)
    board = board.copy()

    board[tuple(box)] = TYPE_LOOKUP['box target'] if board[tuple(box)] == TYPE_LOOKUP['box on target']\
            else TYPE_LOOKUP['empty space']

    board[tuple(player)] = TYPE_LOOKUP['box on target'] if board[tuple(player)] == TYPE_LOOKUP['box target']\
            else TYPE_LOOKUP['box not on target']

    return board


def update_next_state(
    board: np.array,
    available_cells: set[tuple],
    player: (int, int),
    box: np.array,
    visited: set[bytes],
    to_visit: list[tuple],
    macro_states: list[np.array],
    ):
    """Update :visited: boards if needed, and then
    append to :to_visit: and to :macro_states: the new state
    for the next search.
    """
    if board.data.tobytes() in visited:
        return  # We already visited this board

    # Compute neighbours
    available = available_cells.copy()
    available.remove(tuple(box))  # Don't forget to remove the new box position before computations
    neighbours = find_neighbours(player, available, False)

    # Update lists and sets
    to_visit.append((board.copy(), neighbours, box))
    visited.add(board.data.tobytes())
    macro_states.append((board.copy(), player))


def macro_moves(
        board: np.array,
        player: np.array,
        box: np.array,
        forward: bool,
    )-> list[tuple]:
    """Compute all states reachable by moving the given `box`
    by the player.
    Return the list of all states reachable.

    Each state is the result of making mutiple legal move in the board
    by the player when only moving or pushing or pulling the specified box.

    The player can either push or pull according to the :forward: parameter.
    """
    available_cells = np.argwhere(
        (board == TYPE_LOOKUP['empty space']) |\
        (board == TYPE_LOOKUP['box target'])
    )
    available_cells = set(tuple(c) for c in available_cells)
    neighbours = find_neighbours(tuple(player), available_cells, False)
    available_cells.add(tuple(box))  # This set represent all free cells, we omit the current box position

    visited = {board.data.tobytes()}
    to_visit = [(board.copy(), neighbours, box)]  # LIFO
    macro_states = list()

    while to_visit:
        board, neighbours, box = to_visit.pop()
        for player in yield_neighbours(box):
            if forward:
                if not check_push(board, neighbours, player, box):
                    continue  # Illegal move

                n_board = apply_push(board, player, box)
                n_box, player = box + (box - player), box  # New box and player positions
            else:
                if not check_pull(board, neighbours, player, box):
                    continue  # Illegal move

                n_board = apply_pull(board, player, box)
                n_box, player = np.array(player), np.array(player) - (box - player)  # New box and player positions

            update_next_state(
                n_board,
                available_cells,
                tuple(player),
                n_box,
                visited,
                to_visit,
                macro_states,
            )

    return macro_states


if __name__ == '__main__':
    from gym_sokoban.envs import sokoban_env
    from utils import build_board_from_raw, print_board

    env = sokoban_env.SokobanEnv((5, 5), num_boxes=2)
    raw = env.reset(render_mode='raw')

    board, player = build_board_from_raw(raw)
    box = np.argwhere(raw[2] == 1)[0]
    s = macro_moves(board, player, box, True)

    print('MACRO MOVES:')
    print('initial board:')
    print_board(board, player)
    print(f'Box considered: {box}\n\n')
    print(f'Number of macro moves: {len(s)}')
    print('List of macro moves')
    for b, p in s:
        print_board(b, p)
        print('')


    board[board == TYPE_LOOKUP['box not on target']] = TYPE_LOOKUP['empty space']
    board[board == TYPE_LOOKUP['box target']] = TYPE_LOOKUP['box on target']
    box = np.argwhere(board == TYPE_LOOKUP['box on target'])[0]
    s = macro_moves(board, player, box, False)

    print('\n\nMACRO MOVES BACKWARD:')
    print('initial board:')
    print_board(board, player)
    print(f'Box considered: {box}\n\n')
    print(f'Number of macro moves: {len(s)}')
    print('List of macro moves')
    for b, p in s:
        print_board(b, p)
        print('')
