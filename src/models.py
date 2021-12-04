"""Implementation of the main BaseModel and its derivatives.
"""
import numpy as np

import torch
import torch.nn as nn

from features import core_features, all_features


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

    def estimate(self, node):
        """Update the node's value and return the prediction.

        It first computes the features
        and then makes a forward pass.
        """
        raise NotImplementedError


class LinearModel(BaseModel):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)

    def estimate(self, node, gamma: float, backward_solution: list = None):
        """Update the node's value and return the prediction.
        It needs the gamma parameter for the features computations.
        """
        if backward_solution:
            features = all_features(node.env, gamma, backward_solution)
        else:
            features = core_features(node.env, gamma)

        features = torch.FloatTensor(features).unsqueeze(dim=0)
        value = self(features)
        node.value = value[0].item()
        return value
