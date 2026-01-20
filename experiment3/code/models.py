"""
模型定义：支持原始 GAN（MLP）、DCGAN（卷积）、WGAN-GP（沿用 DCGAN 的 critic）
"""
import torch
import torch.nn as nn


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        try:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        except Exception:
            pass
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias.data, 0)


class MLPGenerator(nn.Module):
    def __init__(self, z_dim=100, out_channels=1, img_size=28):
        super().__init__()
        self.img_size = img_size
        self.out_dim = out_channels * img_size * img_size
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, self.out_dim),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.net(z)
        return x.view(z.size(0), -1, self.img_size, self.img_size)


class MLPDiscriminator(nn.Module):
    def __init__(self, in_channels=1, img_size=28):
        super().__init__()
        in_dim = in_channels * img_size * img_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)


# DCGAN-like generator for 28x28 grayscale
class DCGenerator(nn.Module):
    def __init__(self, z_dim=100, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            # input Z, going into a convolution
            nn.ConvTranspose2d(z_dim, 128, 4, 1, 0, bias=False),  # 4x4
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1, bias=False),  # 32x32 (we will crop or adapt)
            nn.Tanh()
        )

    def forward(self, z):
        # z: (B, z_dim)
        z = z.view(z.size(0), z.size(1), 1, 1)
        img = self.net(z)
        # adapt to 28x28 if necessary
        return torch.nn.functional.interpolate(img, size=(28, 28), mode='bilinear', align_corners=False)


class DCDiscriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1, bias=False),  # 14x14
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),  # 7x7
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 3, 1, 0, bias=False),  # 5x5 -> 3x3
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)


# Critic for WGAN-GP (no sigmoid)
class Critic(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, 32, 4, 2, 1)),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Conv2d(32, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 3, 1, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(128 * 5 * 5, 1))
        )

    def forward(self, x):
        return self.net(x).view(-1)
