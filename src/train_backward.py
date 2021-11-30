import torch
import wandb as wb
from tqdm.auto import tqdm, trange  # Loading bar

import environments
from environments import MacroSokobanEnv
from train import train_on_env
from models import LinearModel
from features import core_features
from variables import MAX_MICROSOKOBAN


def train_backward(config: dict) -> LinearModel:
    """Train a basic linear model on all the backward tasks.
    It is trained on the MicroSokoban levels.
    One epoch is a passage through all the levels (155).

    Return the trained linear model.
    """
    torch.manual_seed(config['seed'])
    env = MacroSokobanEnv(forward=False, dim_room=(6, 6), num_boxes=2)
    feat_size = len(core_features(env, config['gamma']))
    model = LinearModel(feat_size)
    config['optimizer'] = torch.optim.Adam(model.parameters(), lr=config['lr'])

    wb.config = config
    wb.watch(model)

    for e in trange(config['epochs'], desc='Epochs'):
        total_loss = 0
        for level_id in trange(1, MAX_MICROSOKOBAN+1, desc='Levels'):
            env = environments.from_file(
                'MicroSokoban',
                level_id,
                forward=False,
                max_steps=config['max_steps'],
            )
            _, loss = train_on_env(model, env, config)

            total_loss += loss

        wb.log({
            'loss': total_loss,
        })

    return model


if __name__ == '__main__':
    config = {
        'gamma': 0.9,
        'max_steps': 120,
        'epsilon': 0.1,
        'seed': 0,
        'epochs': 10,
        'lr': 1e-3,
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
