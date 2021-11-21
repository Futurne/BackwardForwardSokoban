"""Implementation of the main BaseModel and its derivatives.
"""
import numpy as np

import torch
import torch.nn as nn

from utils import core_feature


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

    def estimate(self, node, gamma: float):
        features = core_feature(node.env, gamma)
        features = torch.FloatTensor(features).unsqueeze(dim=0)
        value = self(features)
        node.value = value[0].item()
        return value
