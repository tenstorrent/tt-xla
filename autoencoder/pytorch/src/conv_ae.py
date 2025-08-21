# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch


# SPDX-FileCopyrightText: Copyright (c) 2018 Udacity
#
# SPDX-License-Identifier: MIT
# https://github.com/udacity/deep-learning-v2-pytorch


class ConvAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder_conv2d_1 = torch.nn.Conv2d(1, 16, 3, padding=1)
        self.encoder_conv2d_2 = torch.nn.Conv2d(16, 4, 3, padding=1)
        self.encoder_max_pool2d = torch.nn.MaxPool2d(2, 2)

        # Decoder
        self.decoder_conv2d_1 = torch.nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.decoder_conv2d_2 = torch.nn.ConvTranspose2d(16, 1, 2, stride=2)

        # Activation Function
        self.act_fun = torch.nn.ReLU()

    def forward(self, x):
        # Encode
        act = self.encoder_conv2d_1(x)
        act = self.act_fun(act)
        act = self.encoder_max_pool2d(act)
        act = self.encoder_conv2d_2(act)
        act = self.act_fun(act)
        act = self.encoder_max_pool2d(act)

        # Decode
        act = self.decoder_conv2d_1(act)
        act = self.act_fun(act)
        act = self.decoder_conv2d_2(act)

        return act
