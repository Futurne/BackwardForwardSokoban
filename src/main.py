import torch
from torch.optim.lr_scheduler import StepLR

import wandb as wb
from numpy.random import default_rng
from tqdm import tqdm, trange  # Loading bar

import environments
from environments import MacroSokobanEnv
from train import train_on_env, eval_on_env, train_on_solution
from models import LinearModel
from variables import MAX_MICROSOKOBAN

# MAX_MICROSOKOBAN = 10


def train_backward(config: dict) -> LinearModel:
    """Train a basic linear model on all the backward tasks.
    It is trained on the MicroSokoban levels.
    One epoch is a passage through all the levels (155).

    Return the trained linear model.

    Args
    ----
        :config:    All the needed variables for the training algorithm.
        Those are:
            :gamma:     Gamma coefficient for the RL expected reward computations.
            :max_steps: Number of nodes the search tree can have.
            :epsilon:   For exploration.
            :seed:      Random seed for reproductibility.
            :lr:        Learning rate.
            :epochs:    Number of epochs.
    """
    torch.manual_seed(config['seed'])
    rng = default_rng(config['seed'])
    levels = [l for l in range(1, MAX_MICROSOKOBAN+1)]

    model = LinearModel(n_features=5)
    if config['reload']:
        model.load_state_dict(torch.load('backward.pth'))
    model.train()

    config['optimizer'] = torch.optim.SGD(model.parameters(), lr=config['lr'])
    optimizer = config['optimizer']
    scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

    wb.config = config
    wb.watch(model)

    for e in trange(config['epochs'], desc='Epochs'):
        total_loss = 0
        rng.shuffle(levels)
        winrate = 0
        mean_steps = 0
        for level_id in trange(MAX_MICROSOKOBAN, desc='Levels'):
            env = environments.from_file(
                'MicroSokoban',
                levels[level_id],
                forward=False,
                max_steps=config['max_steps'],
            )
            solution, loss = train_on_env(model, env, config)

            # Keep trying while we lose and while there is another starting position to try
            while not solution[-1].env._check_if_won() and env.starting_pos:
                env.try_again()
                solution, loss = train_on_env(model, env, config)

            total_loss += loss

            # Replay buffer
            loss = train_on_solution(model, solution, config, None)
            total_loss += loss

            mean_steps += len(solution) if solution[-1].env._check_if_won() else config['max_steps']
            winrate += int(solution[-1].env._check_if_won())

            torch.save(model.state_dict(), 'backward.pth')

        scheduler.step()

        wb.log({
            'loss': total_loss,
            'mean steps': mean_steps / MAX_MICROSOKOBAN,
            'winrate': winrate / MAX_MICROSOKOBAN,
        })

    return model


def train_forward(config: dict) -> LinearModel:
    """Train a basic linear model on all the forward tasks.
    It is trained on the MicroSokoban levels.
    One epoch is a passage through all the levels (155).

    Among all the needed variables, it needs the backward trained agent.

    Return the trained linear model.

    Args
    ----
        :config:    All the needed variables for the training algorithm.
        Those are:
            :backward_model:    The model trained in backward mode.
            :gamma:             Gamma coefficient for the RL expected reward computations.
            :max_steps:         Number of nodes the search tree can have.
            :epsilon:           For exploration.
            :seed:              Random seed for reproductibility.
            :lr:                Learning rate.
            :epochs:            Number of epochs.
    """
    torch.manual_seed(config['seed'])
    rng = default_rng(config['seed'])
    levels = [l for l in range(1, MAX_MICROSOKOBAN+1)]

    model = LinearModel(n_features=7)
    if config['reload']:
        model.load_state_dict(torch.load('forward.pth'))
    model.train()

    backward_model = config['backward_model']
    backward_model.eval()

    config['optimizer'] = torch.optim.SGD(model.parameters(), lr=config['lr'])
    optimizer = config['optimizer']
    scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

    wb.config = config
    wb.watch(model)

    for e in trange(config['epochs'], desc='Epochs'):
        total_loss = 0
        rng.shuffle(levels)
        winrate = 0
        mean_steps = 0
        for level_id in trange(MAX_MICROSOKOBAN, desc='Levels'):
            # BACWARD MODE
            env = environments.from_file(
                'MicroSokoban',
                levels[level_id],
                forward=False,
                max_steps=config['max_steps'],
            )
            solution, _ = eval_on_env(backward_model, env, config)

            # Keep trying while we lose and while there is another starting position to try
            while not solution[-1].env._check_if_won() and env.starting_pos:
                env.try_again()
                solution, _ = eval_on_env(backward_model, env, config)


            # FORWARD MODE
            backward_sol = solution
            env = environments.from_file(
                'MicroSokoban',
                levels[level_id],
                forward=True,
                max_steps=config['max_steps'],
            )
            solution, loss = train_on_env(model, env, config, backward_sol=backward_sol)
            total_loss += loss

            # Replay buffer
            loss = train_on_solution(model, solution, config, backward_sol)
            total_loss += loss

            mean_steps += len(solution) if solution[-1].env._check_if_won() else config['max_steps']
            winrate += int(solution[-1].env._check_if_won())

            torch.save(model.state_dict(), 'forward.pth')

        scheduler.step()

        wb.log({
            'loss': total_loss,
            'mean steps': mean_steps / MAX_MICROSOKOBAN,
            'winrate': winrate / MAX_MICROSOKOBAN,
        })

    return model


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2 or sys.argv[1] not in ('backward', 'forward'):
        print(f'Usage: python3 {sys.argv[0]} [backward|forward] [reload]')
        sys.exit(0)

    reload_model = False
    if len(sys.argv) == 3:
        if sys.argv[2] == 'reload':
            reload_model = True
            print('Reloading the model')
        else:
            print(f'Usage: python3 {sys.argv[0]} [backward|forward] [reload]')
            sys.exit(0)

    if sys.argv[1] == 'backward':
        config = {
            'gamma': 0.99,
            'max_steps': 50,
            'epsilon': 0.1,
            'seed': 0,
            'epochs': 100,
            'lr': 1e-2,
            'reload': reload_model,
        }

        with wb.init(
                project='sokoban',
                entity='pierrotlc',
                group='backward training',
                config=config,
                save_code=True,
            ):
            model = train_backward(config)
            torch.save(model.state_dict(), 'backward.pth')

    elif sys.argv[1] == 'forward':
        backward_model = LinearModel(n_features=5)
        backward_model.load_state_dict(torch.load('../models/backward.pth'))

        config = {
            'backward_model': backward_model,
            'gamma': 0.99,
            'max_steps': 100,
            'epsilon': 0.01,
            'seed': 0,
            'epochs': 100,
            'lr': 1e-3,
            'reload': reload_model,
        }

        with wb.init(
                project='sokoban',
                entity='pierrotlc',
                group='forward training',
                config=config,
                save_code=True,
            ):
            model = train_forward(config)
            torch.save(model.state_dict(), 'forward.pth')
