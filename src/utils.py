"""Bunch of utils functions for the game of Sokoban.
"""
import numpy as np

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


def yield_neighbours(cell: (int, int)):
    """Yields all the neighbours around the cell.
    """
    x, y = cell
    for n in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
        yield n


def build_board_from_raw(raw_board: np.array) -> (np.array, np.array):
    """Merge the differents layers of the raw board into a single 2D board.
    Also returns the player coordinates.

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

    player_coords = np.argwhere(player == 1)[0]
    return board, player_coords


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
    available_cells = set(tuple(c) for c in available_cells)
    neighbours = []

    while available_cells:
        current_cell = available_cells.pop()
        n = find_neighbours(current_cell, available_cells, True)
        neighbours.append(set(n))

    return neighbours


def targets(board: np.array) -> int:
    """
    Return the number of boxes already packed on
    target square.
    """
    return env.boxes_on_target


def distance(board: np.array) -> int:
    """
    Return the total distance of the boxes from the targets.
    This is a lower bound of the number of moves required.
    """
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

    return total_distance


def gamma1(board: np.array, gamma: float) -> int:
    """
    A crude estimate of the reward for solving the level.
    """
    return gamma**env.num_boxes


def gamma2(board: np.array, gamma: float) -> int:
    """
    A refined version of gamma1, taking into account the boxes already on target.
    """
    return gamma**(env.num_boxes-env.boxes_on_target)


def core_feature(board, gamma) -> list[float]:
    """
    Return a list of all the core features.
    """
    return [targets(board), distance(board), gamma1(board, gamma), gamma2(board, gamma), connectivity(board)]

def param_env_from_bord(board: np.array):
    """
    Return dim_room, max_steps, num_boxes, num_gen_steps in order to create
    an environment adapted to the levels XSokoban and MicroSokoban using MacroSokobanEnv
    """
    
    dim_room = board.shape
    max_steps = 120
    num_boxes =  np.count_nonzero(board == TYPE_LOOKUP['box target'])
    num_gen_steps= None
    
    return dim_room, max_steps, num_boxes, num_gen_steps

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
    }
    
    file1 = open('levels/XSokoban/screen.'+str(num_lvl), 'r')
    Lines = file1.readlines()
    height, width = len(Lines),max([len(Lines[k]) for k in range(len(Lines))])-1
    board = np.zeros((height, width))
    k=0
    for line in Lines:
        L=[]
        for elt in line:
            for key,val in TYPE_LOOKUP_XSOKOBAN.items():
                if elt == val:
                    L.append(TYPE_LOOKUP[key])
        board[k]=L+[TYPE_LOOKUP['wall'] for k in range(width-len(L))]
        k+=1
    return board


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
    }
    
    file1 = open('levels/MicroSokoban/screen.'+str(num_lvl), 'r')
    Lines = file1.readlines()
    height, width = len(Lines),max([len(Lines[k]) for k in range(len(Lines))])-1
    board = np.zeros((height, width))
    k=0
    for line in Lines:
        L=[]
        for elt in line:
            for key,val in TYPE_LOOKUP_XSOKOBAN.items():
                if elt == val:
                    L.append(TYPE_LOOKUP[key])
        board[k]=L+[TYPE_LOOKUP['wall'] for k in range(width-len(L))]
        k+=1
    return board


if __name__ == '__main__':
    from gym_sokoban.envs import sokoban_env
    env = sokoban_env.SokobanEnv((10, 10), num_boxes=2)
    raw = env.reset(render_mode='raw')
    board, _ = build_board_from_raw(raw)
    print_board(board)
    n = connectivity(board)
    t = targets(board)
    d = distance(board)
    g1 = gamma1(board, 0.99)
    g2 = gamma1(board, 0.99)
    print("\nConnectivity: {}\nBoxes on target: {}\nDistance: {}\nGamma1: {}\nGamma2: {}".format(len(n), t, d, g1, g2))
