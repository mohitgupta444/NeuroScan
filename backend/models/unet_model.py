"""
U-Net for Brain Tumor Segmentation
------------------------------------
A compact U-Net built from scratch with a small number of channels so it
trains and runs comfortably on CPU / 8GB RAM. This is intentionally a
lighter U-Net than the classic 64-128-256-512-1024 channel design, because
we are training on pseudo-masks (see generate_pseudo_masks.py), not
hand-annotated masks, and a smaller model is faster to iterate on and
less likely to overfit noisy pseudo-labels.

Input:  1-channel grayscale MRI slice, 256x256
Output: 1-channel probability mask (sigmoid), 256x256
"""

import torch
import torch.nn as nn


def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class LightUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 16):
        super().__init__()

        # ---- Encoder ----
        self.enc1 = conv_block(in_channels, base)          # 256 -> 256
        self.enc2 = conv_block(base, base * 2)              # 128 -> 128
        self.enc3 = conv_block(base * 2, base * 4)          # 64  -> 64
        self.enc4 = conv_block(base * 4, base * 8)          # 32  -> 32
        self.pool = nn.MaxPool2d(2)

        # ---- Bottleneck ----
        self.bottleneck = conv_block(base * 8, base * 16)   # 16  -> 16

        # ---- Decoder (with skip connections, the defining U-Net trait) ----
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = conv_block(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = conv_block(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)

        self.out_conv = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))


if __name__ == "__main__":
    # quick shape sanity check
    m = LightUNet()
    x = torch.randn(2, 1, 256, 256)
    y = m(x)
    print("output shape:", y.shape)  # expect [2, 1, 256, 256]
    n_params = sum(p.numel() for p in m.parameters())
    print(f"params: {n_params/1e6:.2f}M")
