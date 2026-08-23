import torch
import torch.nn as nn

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return residual + x

class ConvNeXtSpatialEncoder(nn.Module):
    def __init__(self, in_channels: int = 6, dims: list = [64, 128, 256]):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(1, dims[0])
        )
        self.stage1 = nn.Sequential(ConvNeXtBlock(dims[0]), ConvNeXtBlock(dims[0]))
        self.down1 = nn.Sequential(nn.GroupNorm(1, dims[0]), nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2))
        self.stage2 = nn.Sequential(ConvNeXtBlock(dims[1]), ConvNeXtBlock(dims[1]))
        self.down2 = nn.Sequential(nn.GroupNorm(1, dims[1]), nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2))
        self.stage3 = nn.Sequential(ConvNeXtBlock(dims[2]), ConvNeXtBlock(dims[2]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        return x
