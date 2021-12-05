"""Bunch of utils functions for the game of Sokoban.
"""
import numpy as np
from numba import njit, jit

from variables import TYPE_LOOKUP


def print_board(board: np.array, player: np.array=None):
    """Print the board in a human-readable way.
    """
    if player is not None:
        coords_player = tuple(player)

    for x in range(len(board)):
        for y in range(len(board[0])):
            if player is not None and coords_player == (x, y):
                print(TYPE_LOOKUP['player'], end=' ')
            elif board[x, y] == TYPE_LOOKUP['empty space']:
                print(' ', end=' ')
            else:
                print(board[x, y], end=' ')
        print('\n', end='')


@njit(inline='always')
def yield_neighbours(cell: (int, int)) -> list:
    """Yields all the neighbours around the cell.
    """
    x, y = cell
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]


def build_board_from_raw(raw_board: np.array) -> (np.array, np.array):
    """Merge the differents layers of the raw board into a single 2D board.
    Also returns the player coordinates.

    The number corresponding of each type of cell is given by `variables.TYPE_LOOKUP`.
    """
    walls, goals, boxes, player = raw_board
    board = ((walls | goals | boxes) == 0) * TYPE_LOOKUP['empty space']
    board += (walls == 1) * TYPE_LOOKUP['wall']
    board += (goals & boxes) * TYPE_LOOKUP['box on target']  # Find boxes and target
    board += ((goals ^ boxes) & goals) * TYPE_LOOKUP['box target']  # Find target without boxes
    board += ((boxes ^ goals) & boxes) * TYPE_LOOKUP['box not on target']  # Find boxes without target

    player_coords = np.argwhere(player == 1)
    if len(player_coords) != 0:
        player_coords = player_coords[0]
    return board, player_coords


@jit(nopython=False)
def find_neighbours(cell: tuple, available_cells: set[tuple], remove_cells: bool) -> list[tuple]:
    """Find all neighbours in the `available_cells`.

    If the :remove_cells: parameter is True, then all found neighbours
    will be removed from :available_cells:.
    Return those neighbours.
    """
    neighbours = [cell]
    to_visit = [cell]  # LIFO

    if not remove_cells:
        available_cells = available_cells.copy()  # Do not modify original set

    while to_visit:
        cell = to_visit.pop()
        for n in yield_neighbours(cell):
            if n in available_cells:
                neighbours.append(n)
                to_visit.append(n)
                available_cells.remove(n)

    return neighbours


def connectivity(env) -> list[set]:
    """Find all group of cells that are freely walkable
    by the player.

    Return a list of set of cells. Each set contains
    the cells that can be reach by each other.
    """
    raw = env.render()
    board, _ = build_board_from_raw(raw)
    available_cells = np.argwhere(
        (board == TYPE_LOOKUP['empty space']) |\
        (board == TYPE_LOOKUP['box target'])
    )
    # Transforms the numpy array into a set of tuples
    available_cells = set(tuple(c) for c in available_cells)
    neighbours = []

    while available_cells:
        current_cell = available_cells.pop()
        n = find_neighbours(current_cell, available_cells, True)
        neighbours.append(set(n))

    return neighbours


def is_env_deadlock(env) -> bool:
    """Return whether the board is a trivial deadlock or not.
    A trivial deadlock is a room where every accessible box cannot be pushed
    and where the game is not finished.
    """
    if env._check_if_done():
        return False  # The game is finished => no deadlocks

    return len(env.reachable_states()) == 0  # No possible moves


def XSokoban_lvl_to_raw(num_lvl:int) -> np.array:
    """
    Return board in raw format of a level of XSokoban
    XSokoban levels bank from :
    https://www.cs.cornell.edu/andru/xsokoban.html
    """
    TYPE_LOOKUP_XSOKOBAN = {
        'wall': '#',
        'empty space': ' ',
        'box target': '.',
        'box on target': '*',
        'box not on target': '$',
        'player': '@',
        'player on target': '+',
    }

    with open('../levels/XSokoban/screen.' + str(num_lvl), 'r') as file1:
        Lines = file1.readlines()
    height, width = len(Lines),max([len(Lines[k]) for k in range(len(Lines))])-1
    board = np.zeros((height, width), int)
    k=0
    for line in Lines:
        L=[]
        for elt in line:
            # Check for player coords
            if elt == TYPE_LOOKUP_XSOKOBAN['player on target']:
                elt = TYPE_LOOKUP_XSOKOBAN['box target']
                player = np.array([k, len(L)])
            elif elt == TYPE_LOOKUP_XSOKOBAN['player']:
                elt = TYPE_LOOKUP_XSOKOBAN['empty space']
                player = np.array([k, len(L)])

            for key,val in TYPE_LOOKUP_XSOKOBAN.items():
                if elt == val:
                    L.append(TYPE_LOOKUP[key])
        board[k]=L+[TYPE_LOOKUP['wall'] for k in range(width-len(L))]
        k+=1
    return board, player


def MicroSokoban_lvl_to_raw(num_lvl:int) -> np.array:
    """
    Return board in raw format of a level of MicroSokoban
    MicroSokoban levels bank from :
    http://www.abelmartin.com/rj/sokobanJS/Skinner/David%20W.%20Skinner%20-%20Sokoban_files/Microban.txt
    """
        
    TYPE_LOOKUP_XSOKOBAN = {
        'wall': '#',
        'empty space': ' ',
        'box target': '.',
        'box on target': '*',
        'box not on target': '$',
        'player': '@',
        'player on target': '+',
    }
    
    file1 = open('../levels/MicroSokoban/screen.'+str(num_lvl), 'r')
    Lines = file1.readlines()
    height, width = len(Lines),max([len(Lines[k]) for k in range(len(Lines))])-1
    board = np.zeros((height, width), int)
    k=0
    for line in Lines:
        L=[]
        for elt in line:
            # Check for player coords
            if elt == TYPE_LOOKUP_XSOKOBAN['player on target']:
                elt = TYPE_LOOKUP_XSOKOBAN['box target']
                player = np.array([k, len(L)])
            elif elt == TYPE_LOOKUP_XSOKOBAN['player']:
                elt = TYPE_LOOKUP_XSOKOBAN['empty space']
                player = np.array([k, len(L)])

            for key,val in TYPE_LOOKUP_XSOKOBAN.items():
                if elt == val:
                    L.append(TYPE_LOOKUP[key])
        board[k]=L+[TYPE_LOOKUP['wall'] for k in range(width-len(L))]
        k+=1
    return board, player


if __name__ == '__main__':
    from gym_sokoban.envs import sokoban_env
    env = sokoban_env.SokobanEnv((10, 10), num_boxes=2)
    raw = env.reset(render_mode='raw')
    board, _ = build_board_from_raw(raw)
    print_board(board)
    features = core_feature(env, 0.99)
    print("\nConnectivity: {}\nBoxes on target: {}\nDistance: {}\nGamma1: {}\nGamma2: {}".format(features[0], features[1], features[2], features[3], features[4]))
