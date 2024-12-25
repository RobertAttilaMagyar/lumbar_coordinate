import torch
import torch.nn as nn


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
        dropout: float = 0.0,
        with_batchnorm: bool = False,
    ):
        dout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=not (with_batchnorm),
            padding=padding,
        )
        bn = nn.BatchNorm2d(out_channels) if with_batchnorm else nn.Identity()
        relu = nn.ReLU(inplace=True)

        super(Conv2dReLU, self).__init__(dout, conv, bn, relu)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0):
        super().__init__()
        self._dc = nn.ModuleList()
        for _ in range(2):
            if dropout != 0:
                self._dc.append(nn.Dropout(p=dropout))
            self._dc.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                )
            )
            in_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._dc:
            x = layer(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None):
        if out_channels is None:
            out_channels = 2 * in_channels
        super().__init__()
        self._double_conv = DoubleConv(in_channels, out_channels, dropout=0.2)
        self._max_pool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self._double_conv(x)
        out2 = self._max_pool(out1)
        return out1, out2


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int = 512):
        super().__init__()
        self._up_conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=2,
            stride=2,
        )
        self._double_conv = DoubleConv(in_channels, in_channels // 2)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = self._up_conv(x)
        out = torch.cat((y, x), dim=1)
        out = self._double_conv(out)
        return out


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self._seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self._seq(x)


class UNet(nn.Module):
    def __init__(self, num_blocks: int = 3, input_dims: tuple[int, int] = (256, 256)):
        super().__init__()
        self._encoders = nn.ModuleList()
        self._decoders = nn.ModuleList()
        for i in range(num_blocks):
            if i == 0:
                self._encoders.append(EncoderBlock(1, 64))
            else:
                self._encoders.append(EncoderBlock(in_channels=2 ** (i - 1) * 64))

            self._decoders.append(DecoderBlock(2 ** (num_blocks - i) * 64))

        self._bottleneck = BottleNeck(
            in_channels=2 ** (num_blocks - 1) * 64, out_channels=(2**num_blocks) * 64
        )

        self._decode_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

        self._linear = nn.Linear((256 * 256), out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for_skip: list[torch.Tensor] = []
        for enc in self._encoders:
            tmp, x = enc(x)
            for_skip.append(tmp)
        x = self._bottleneck(x)
        for idx, dec in enumerate(self._decoders):
            x = dec(x, for_skip[len(for_skip) - 1 - idx])

        x = self._decode_conv(x)
        x = x.view(-1, 256 * 256)
        x = self._linear(x)
        x = x.view(-1, 5, 2)
        return x
