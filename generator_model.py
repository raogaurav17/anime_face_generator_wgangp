import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim=100, channel_img=3, feature_g=64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self.block(z_dim, feature_g * 16, 4, 1, 0),  # 1x1 → 4x4
            self.block(feature_g * 16, feature_g * 8, 4, 2, 1),  # 4x4 → 8x8
            self.block(feature_g * 8, feature_g * 4, 4, 2, 1),  # 8x8 → 16x16
            self.block(feature_g * 4, feature_g * 2, 4, 2, 1),  # 16x16 → 32x32
            nn.ConvTranspose2d(
                feature_g * 2,
                channel_img,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),  # 32x32 → 64x64
            nn.Tanh(),
        )

    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)
