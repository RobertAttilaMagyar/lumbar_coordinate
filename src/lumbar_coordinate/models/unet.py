import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self._double_conv = nn.Sequential(
            nn.Dropout(p = 0.1, inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._double_conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None):
        if out_channels is None:
            out_channels = 2 * in_channels
        super().__init__()
        self._double_conv = DoubleConv(in_channels, out_channels)
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

            self._decoders.append(DecoderBlock(2**(num_blocks - i) * 64))

        self._bottleneck = nn.Conv2d(
            in_channels=2 ** (num_blocks - 1) * 64,
            out_channels=(2**num_blocks) * 64,
            kernel_size=3,
            padding=1,
        )

        self._decode_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

        self._linear = nn.Linear((256 * 256), out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for_skip:list[torch.Tensor]  = []
        for enc in self._encoders:
            tmp, x = enc(x)
            for_skip.append(tmp)
        x = self._bottleneck(x)
        for idx, dec in enumerate(self._decoders):
            x = dec(x, for_skip[len(for_skip) - 1 - idx])
        
        x = self._decode_conv(x)
        x = x.view(-1, 256*256)
        x = self._linear(x)
        x = x.view(-1, 5,2)
        return x

dummy_input = torch.rand(10,1,256,256)
unet = UNet(num_blocks = 3, input_dims = tuple(dummy_input.shape)[2:])
dummy_output = unet(dummy_input)
print(dummy_output.shape)
# print(unet)


