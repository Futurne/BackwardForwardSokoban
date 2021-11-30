import torch
from tqdm import tqdm  # Loading bar

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
    """
    env = MacroSokobanEnv(forward=False, dim_room=(6, 6), num_boxes=2)
    feat_size = len(core_features(env, config['gamma']))
    model = LinearModel(feat_size)
    config['optimizer'] = torch.optim.Adam(model.parameters(), lr=config['lr'])

    for _ in tqdm(range(config['epochs'])):
        for level_id in tqdm(range(1, MAX_MICROSOKOBAN+1)):
            env = environments.from_file(
                'MicroSokoban',
                level_id,
                forward=False,
                max_steps=config['max_steps'],
            )
            train_on_env(model, env, config)

    return model


if __name__ == '__main__':
    config = {
        'gamma': 0.9,
        'max_steps': 120,
        'epsilon': 0.1,
        'seed': 0,
        'epochs': 2,
        'lr': 1e-4,
    }

    model = train_backward(config)
