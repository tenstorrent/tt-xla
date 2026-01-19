# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv. Uses bilinear upsample by default."""

    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        # in_ch: number of channels coming from previous decoder (after concat)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, mid_ch=in_ch // 2)
        else:
            # transpose conv
            self.up = nn.ConvTranspose2d(
                in_ch // 2, in_ch // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        """
        x1: decoder feature (coarser)
        x2: encoder skip feature (finer) - already gated before call in AttentionUNet
        """
        x1 = self.up(x1)
        # Pad x1 to x2 size if needed (due to rounding)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffY != 0 or diffX != 0:
            x1 = F.pad(
                x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate from Attention U-Net (Oktay et al.)
    It receives:
      - x (encoder feature)  => the feature to be gated (higher resolution)
      - g (decoder feature)  => the gating signal (coarser)
    Produces: gated x (same shape as x)
    """

    def __init__(self, in_channels_x, in_channels_g, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(1, in_channels_x // 2)

        # theta_x: transform encoder feature to intermediate space
        self.theta_x = nn.Sequential(
            nn.Conv2d(
                in_channels_x,
                inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(inter_channels, momentum=0.01, eps=1e-3),
        )
        # phi_g: transform gating feature to intermediate space
        self.phi_g = nn.Sequential(
            nn.Conv2d(
                in_channels_g,
                inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(inter_channels, momentum=0.01, eps=1e-3),
        )
        # psi: produce attention map
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x, g):
        """
        x: encoder tensor (B, Cx, H, W)
        g: decoder tensor (B, Cg, Hg, Wg)
        returns: x * attention_map  (same shape as x)
        """
        # transform
        theta_x = self.theta_x(x)  # (B, inter, H, W)
        phi_g = self.phi_g(g)  # (B, inter, Hg, Wg)
        # upsample phi_g to theta_x spatial size if needed
        if phi_g.shape[2:] != theta_x.shape[2:]:
            phi_g = F.interpolate(
                phi_g, size=theta_x.shape[2:], mode="bilinear", align_corners=True
            )
        f = theta_x + phi_g  # broadcasting sum
        psi = self.psi(f)  # (B,1,H,W) in (0,1)
        return x * psi


class AttentionUNet(nn.Module):
    """
    Attention U-Net:
      - Encoder: DoubleConv blocks with downsampling
      - Decoder: Upsample + AttentionGate on skip connections + DoubleConv
    Params:
      n_channels: input channels (e.g., 3)
      n_classes: output channels (e.g., 1 for binary seg)
      features: list of feature sizes for encoder (e.g., [64, 128, 256, 512])
    """

    def __init__(
        self, n_channels=3, n_classes=1, features=[64, 128, 256, 512], bilinear=True
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder (first conv is not pooled)
        self.inc = DoubleConv(n_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])

        # Bottleneck
        self.bottleneck = DoubleConv(features[3], features[3] * 2)

        # Decoder ups: note the in_ch for Up is (dec_feat + enc_feat_after_gating)
        self.up3 = Up(
            in_ch=features[3] * 2 + features[3], out_ch=features[3], bilinear=bilinear
        )
        # 1024 + 512 = 1536

        self.att3 = AttentionGate(
            in_channels_x=features[3],
            in_channels_g=features[3] * 2,
            inter_channels=features[3] // 2,
        )

        self.up2 = Up(features[3] + features[2], features[2])

        self.att2 = AttentionGate(
            in_channels_x=features[2],
            in_channels_g=features[3],
            inter_channels=features[2] // 2,
        )

        self.up1 = Up(features[2] + features[1], features[1])

        self.att1 = AttentionGate(
            in_channels_x=features[1],
            in_channels_g=features[2],
            inter_channels=features[1] // 2,
        )

        self.up0 = Up(features[1] + features[0], features[0])

        self.att0 = AttentionGate(
            in_channels_x=features[0],
            in_channels_g=features[1],
            inter_channels=features[0] // 2,
        )

        # Final conv
        self.outc = nn.Conv2d(features[0], n_classes, kernel_size=1)

        # initialize weights
        self._init_weights()

    def forward(self, x):
        # Encoder
        x0 = self.inc(x)  # features[0]
        x1 = self.down1(x0)  # features[1]
        x2 = self.down2(x1)  # features[2]
        x3 = self.down3(x2)  # features[3]

        # Bottleneck
        x4 = self.bottleneck(x3)  # features[3]*2

        # Decoder with attention gating on encoder skips
        g3 = x4
        x3_att = self.att3(x3, g3)
        x = self.up3(g3, x3_att)  # up from bottleneck, concat gated skip

        g2 = x
        x2_att = self.att2(x2, g2)
        x = self.up2(g2, x2_att)

        g1 = x
        x1_att = self.att1(x1, g1)
        x = self.up1(g1, x1_att)

        g0 = x
        x0_att = self.att0(x0, g0)
        x = self.up0(g0, x0_att)

        logits = self.outc(x)
        return logits

    def _init_weights(self):
        # Kaiming init for convs (common practice)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
