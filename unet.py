import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DoubleConv(nn.Module):
    """(Convolution => [Batch Normalization] => ReLU) * 2 block."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        """
        Initializes the DoubleConv block.
        Uses a 7x7 kernel for the first convolution and 3x3 for the second.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            mid_channels: Number of intermediate channels. Defaults to out_channels.
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block: MaxPool followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes the Down block.

        Args:
            in_channels: Number of input channels to DoubleConv.
            out_channels: Number of output channels from DoubleConv.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block: Upsample/ConvTranspose followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        """
        Initializes the Up block.

        Args:
            in_channels: Number of input channels (from the layer below and skip connection).
            out_channels: Number of output channels from DoubleConv.
            bilinear: If True, use bilinear upsampling, otherwise use ConvTranspose2d.
        """
        super().__init__()

        # If bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Applies upscaling and concatenation with skip connection.

        Args:
            x1: Input tensor from the upsampling path (lower resolution).
            x2: Input tensor from the skip connection (higher resolution).

        Returns:
            Output tensor after upsampling, concatenation, and double convolution.
        """
        x1 = self.up(x1)
        # Input is CHW
        # Calculate padding needed to match dimensions if ConvTranspose2d is used
        # or if input sizes are not perfectly divisible by 2
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # Pad x1 to match the spatial dimensions of x2
        x1 = F.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution block: 1x1 convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes the OutConv block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (classes).
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet architecture for image segmentation or restoration.

    Args:
        n_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB).
        n_classes: Number of output channels (e.g., 1 for regression/binary seg, C for C-class seg).
        bilinear: Whether to use bilinear upsampling in the decoder path. Default is False (uses ConvTranspose2d).
    """

    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        # Encoder path
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)  # Adjusted for bilinear

        # Decoder path
        self.up1 = Up(1024, 512 // factor, bilinear)  # Adjusted for bilinear
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
