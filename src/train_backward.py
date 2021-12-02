import torch
from torch.optim.lr_scheduler import StepLR

import wandb as wb
from numpy.random import default_rng
from tqdm import tqdm, trange  # Loading bar

import environments
from environments import MacroSokobanEnv
from train import train_on_env
from models import LinearModel
from features import core_features
from variables import MAX_MICROSOKOBAN

MAX_MICROSOKOBAN = 10


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

    env = MacroSokobanEnv(forward=False, dim_room=(6, 6), num_boxes=2)
    feat_size = len(core_features(env, config['gamma']))
    model = LinearModel(feat_size)

    config['optimizer'] = torch.optim.SGD(model.parameters(), lr=config['lr'])
    optimizer = config['optimizer']
    scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

    wb.config = config
    wb.watch(model)

    for e in trange(config['epochs'], desc='Epochs'):
        total_loss = 0
        rng.shuffle(levels)
        for level_id in trange(1, MAX_MICROSOKOBAN+1, desc='Levels'):
            env = environments.from_file(
                'MicroSokoban',
                level_id,
                forward=False,
                max_steps=config['max_steps'],
            )
            _, loss = train_on_env(model, env, config)

            if loss:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.cpu().item()
            else:
                tqdm.write(f'Level error: {level_id}')

        scheduler.step()

        wb.log({
            'loss': total_loss,
        })

    return model


if __name__ == '__main__':
    config = {
        'gamma': 0.9,
        'max_steps': 50,
        'epsilon': 0.1,
        'seed': 0,
        'epochs': 10,
        'lr': 1e-2,
    }

    with wb.init(
            project='test',
            entity='pierrotlc',
            group='backward training',
            config=config,
            save_code=True,
        ):
        model = train_backward(config)
        torch.save(model.state_dict(), 'backward.pth')
