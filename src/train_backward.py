import torch
from tqdm import tqdm  # Loading bar

import environments
from environments import MacroSokobanEnv
from train import train_on_env
from models import LinearModel
from variables import MAX_MICROSOKOBAN


def train_backward(config: dict) -> LinearModel:
    """Train a basic linear model on all the backward tasks.
    It is trained on the MicroSokoban levels.
    One epoch is a passage through all the levels (155).

    Return the trained linear model.
    """
    env = MacroSokobanEnv(forward=False, dim_room=(6, 6), num_boxes=2)
    feat_size = len(core_features(env, config['gamma']))
    model = LinearModel(feat_size)
    optim = torch.optim.Adam(model.parameters(), lr=config['lr'])

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
