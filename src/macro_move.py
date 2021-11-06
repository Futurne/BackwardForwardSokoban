"""Utils function to generate macro moves.

A macro-move is a combination of legal moves of a unique box.
Theses macro-moves are equivalent to a multiple serie of movements,
but are much more efficient for the search tree.
"""
import numpy as np

from utils import find_neightbours, yield_neightbours, print_board
from variables import TYPE_LOOKUP


FORBIDDEN = {
    TYPE_LOOKUP['wall'],
    TYPE_LOOKUP['box on target'],
    TYPE_LOOKUP['box not on target'],
}


def check_move(
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


def apply_move(
    board: np.array,
    player: (int, int),
    box: np.array,
    ) -> np.array:
    """Move the box by the player next to it.
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

    # Compute neightbours
    available = available_cells.copy()
    available.remove(tuple(box))  # Don't forget to remove the new box position before computations
    neightbours = find_neightbours(player, available, False)

    # Update lists and sets
    to_visit.append((board.copy(), neightbours, box))
    visited.add(board.data.tobytes())
    macro_states.append((board.copy(), player))


def macro_moves(board: np.array, player: np.array, box: np.array) -> list[tuple]:
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
    available_cells = set(tuple(c) for c in available_cells)
    neightbours = find_neightbours(tuple(player), available_cells, False)
    available_cells.add(tuple(box))  # This set represent all free cells, we omit the current box position

    visited = {board.data.tobytes()}
    to_visit = [(board.copy(), neightbours, box)]  # LIFO
    macro_states = list()

    while to_visit:
        board, neightbours, box = to_visit.pop()
        for player in yield_neightbours(box):
            if not check_move(board, neightbours, player, box):
                continue  # Illegal move

            n_board = apply_move(board, player, box)
            n_box, player = box + (box - player), box  # New box and player positions

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

    s = macro_moves(board, player, box)

    print('MACRO MOVES:')
    print('initial board:')
    print_board(board, player)
    print(f'Box considered: {box}\n\n')
    print(f'Number of macro moves: {len(s)}')
    print('List of macro moves')
    for b, p in s:
        print_board(b, p)
        print('')
