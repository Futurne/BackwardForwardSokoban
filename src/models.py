"""Implementation of the main BaseModel and its derivatives.
"""
import numpy as np

import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Evaluates the batch of boards.

        Args
        ----
            :x: Boards of shape [batch_size, feature_size].

        Returns
        -------
            :v: Values estimated of shape [batch_size, 1].
        """
        raise NotImplementedError

    def estimate(self, board: np.array):
        """Return the value estimated by the model.

        It first computes the features
        and then makes a forward pass.
        """
        raise NotImplementedError
