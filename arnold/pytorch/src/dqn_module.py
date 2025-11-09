# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DQN Module implementations for Arnold - Deep Q-Network for ViZDoom.
"""
import torch
import torch.nn as nn
from logging import getLogger
from .model_utils import (
    build_CNN_network,
    build_game_variables_network,
    build_game_features_network,
    get_recurrent_module,
)


logger = getLogger()


class DQNModuleBase(nn.Module):
    """Base class for DQN modules."""

    def __init__(self, params):
        super(DQNModuleBase, self).__init__()

        # build CNN network
        build_CNN_network(self, params)
        self.output_dim = self.conv_output_dim

        # game variables network
        build_game_variables_network(self, params)
        if self.n_variables:
            self.output_dim += sum(params.variable_dim)

        # dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)

        # game features network
        build_game_features_network(self, params)

        # Estimate state-action value function Q(s, a)
        # If dueling network, estimate advantage function A(s, a)
        self.proj_action_scores = nn.Linear(params.hidden_dim, self.n_actions)

        self.dueling_network = params.dueling_network
        if self.dueling_network:
            self.proj_state_values = nn.Linear(params.hidden_dim, 1)

        # log hidden layer sizes
        logger.info("Conv layer output dim : %i" % self.conv_output_dim)
        logger.info("Hidden layer input dim: %i" % self.output_dim)

    def base_forward(self, x_screens, x_variables):
        """
        Argument sizes:
            - x_screens of shape (batch_size, conv_input_size, h, w)
            - x_variables of shape (batch_size,)
        where for feedforward:
            batch_size == params.batch_size,
            conv_input_size == hist_size * n_feature_maps
        and for recurrent:
            batch_size == params.batch_size * (hist_size + n_rec_updates)
            conv_input_size == n_feature_maps
        Returns:
            - output of shape (batch_size, output_dim)
            - output_gf of shape (batch_size, n_features)
        """
        batch_size = x_screens.size(0)

        # convolution
        x_screens = x_screens / 255.0
        conv_output = self.conv(x_screens).view(batch_size, -1)

        # game variables
        if self.n_variables:
            embeddings = [
                self.game_variable_embeddings[i](x_variables[i])
                for i in range(self.n_variables)
            ]

        # game features
        if self.n_features:
            output_gf = self.proj_game_features(conv_output)
        else:
            output_gf = None

        # create state input
        if self.n_variables:
            output = torch.cat([conv_output] + embeddings, 1)
        else:
            output = conv_output

        # dropout
        if self.dropout:
            output = self.dropout_layer(output)

        return output, output_gf

    def head_forward(self, state_input):
        if self.dueling_network:
            a = self.proj_action_scores(state_input)  # advantage branch
            v = self.proj_state_values(state_input)  # state value branch
            a -= a.mean(1, keepdim=True).expand(a.size())
            return v.expand(a.size()) + a
        else:
            return self.proj_action_scores(state_input)


class DQNModuleFeedforward(DQNModuleBase):
    """
    Deep Q-Network feedforward module for reinforcement learning.

    Architecture:
    - CNN layers for processing screen images
    - Embedding layers for game variables (health, ammo)
    - Feedforward layer
    - Output layer for Q-values
    """

    def __init__(self, params):
        super(DQNModuleFeedforward, self).__init__(params)

        self.feedforward = nn.Sequential(
            nn.Linear(self.output_dim, params.hidden_dim), nn.Sigmoid()
        )

    def forward(self, x_screens, x_variables):
        """
        Argument sizes:
            - x_screens of shape (batch_size, seq_len * n_fm, h, w)
            - x_variables list of n_var tensors of shape (batch_size,)
        """
        batch_size = x_screens.size(0)
        assert x_screens.ndimension() == 4
        assert len(x_variables) == self.n_variables
        assert all(x.ndimension() == 1 and x.size(0) == batch_size for x in x_variables)

        # state input (screen / depth / labels buffer + variables)
        state_input, output_gf = self.base_forward(x_screens, x_variables)

        # apply the feed forward middle
        state_input = self.feedforward(state_input)

        # apply the head to feed forward result
        output_sc = self.head_forward(state_input)

        return output_sc, output_gf


class DQNModuleRecurrent(DQNModuleBase):
    """
    Deep Q-Network recurrent module for reinforcement learning.

    Architecture:
    - CNN layers for processing screen images
    - Embedding layers for game variables (health, ammo)
    - Recurrent layer (RNN/GRU/LSTM)
    - Output layer for Q-values
    """

    def __init__(self, params):
        super(DQNModuleRecurrent, self).__init__(params)

        recurrent_module = get_recurrent_module(params.recurrence)
        self.rnn = recurrent_module(
            self.output_dim,
            params.hidden_dim,
            num_layers=params.n_rec_layers,
            dropout=params.dropout,
            batch_first=True,
        )

    def forward(self, x_screens, x_variables, prev_state):
        """
        Argument sizes:
            - x_screens of shape (batch_size, seq_len, n_fm, h, w)
            - x_variables list of n_var tensors of shape (batch_size, seq_len)
        """
        batch_size = x_screens.size(0)
        seq_len = x_screens.size(1)

        assert x_screens.ndimension() == 5
        assert len(x_variables) == self.n_variables
        assert all(
            x.ndimension() == 2 and x.size(0) == batch_size and x.size(1) == seq_len
            for x in x_variables
        )

        # We're doing a batched forward through the network base
        # Flattening seq_len into batch_size ensures that it will be applied
        # to all timesteps independently.
        state_input, output_gf = self.base_forward(
            x_screens.view(batch_size * seq_len, *x_screens.size()[2:]),
            [v.contiguous().view(batch_size * seq_len) for v in x_variables],
        )

        # unflatten the input and apply the RNN
        rnn_input = state_input.view(batch_size, seq_len, self.output_dim)
        rnn_output, next_state = self.rnn(rnn_input, prev_state)
        rnn_output = rnn_output.contiguous()

        # apply the head to RNN hidden states (simulating larger batch again)
        output_sc = self.head_forward(rnn_output.view(-1, self.hidden_dim))

        # unflatten scores and game features
        output_sc = output_sc.view(batch_size, seq_len, output_sc.size(1))
        if self.n_features:
            output_gf = output_gf.view(batch_size, seq_len, self.n_features)

        return output_sc, output_gf, next_state
