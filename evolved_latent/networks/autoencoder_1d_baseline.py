from typing import Sequence
import torch
import torch.nn as nn
from evolved_latent.networks.network_utils import Activation, _str_to_activation
from evolved_latent.networks.common_blocks import DownResidual1DBlock, UpResidual1DBlock
import numpy as np


class AEBaseline(nn.Module):
    """AEBaseline module for Pytorch."""

    def __init__(
        self,
        input_shape: Sequence[int],
        conv_sizes: Sequence[int],
        linear_sizes: Sequence[int],
        activation: Activation,
    ):
        super().__init__()
        if isinstance(activation, str):
            activation = _str_to_activation[activation]
        encoder_config = {
            "input_shape": input_shape,
            "conv_sizes": conv_sizes,
            "linear_sizes": linear_sizes,
            "activation": activation,
        }
        self.encoder = EncoderBaseline(**encoder_config)

        input_shape = self.encoder(torch.zeros(1, *input_shape)).shape[1:]
        decoder_config = {
            "input_shape": input_shape,
            "conv_sizes": conv_sizes[::-1],
            "linear_sizes": linear_sizes[::-1],
            "activation": activation,
        }
        self.decoder = DecoderBaseline(**decoder_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class EncoderBaseline(nn.Module):
    """EncoderBaseline module for Pytorch."""

    def __init__(
        self,
        input_shape: Sequence[int],
        conv_sizes: Sequence[int],
        linear_sizes: Sequence[int],
        activation: nn.Module,
    ):
        super().__init__()

        input_size = conv_sizes[0]
        self.conv_layers = nn.ModuleList()
        for i in range(1, len(conv_sizes)):
            self.conv_layers.append(
                DownResidual1DBlock(
                    input_size,
                    conv_sizes[i],
                    kernel_size=3,
                    strides=2,
                    padding=1,
                    activation=activation,
                )
            )
            input_size = conv_sizes[i]
        self.conv_layers = nn.Sequential(*self.conv_layers)

        conv_shape = self.conv_layers(torch.zeros(1, *input_shape)).shape[1:]
        input_size = np.prod(conv_shape)

        self.linear_layers = nn.ModuleList()
        for i in range(len(linear_sizes)):
            self.linear_layers.append(nn.Linear(input_size, linear_sizes[i]))
            input_size = linear_sizes[i]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = x.flatten(1)
        for linear_layer in self.linear_layers:
            x = linear_layer(x)

        return x


class DecoderBaseline(nn.Module):
    """DecoderBaseline module for Pytorch."""

    def __init__(
        self,
        input_shape: Sequence[int],
        conv_sizes: Sequence[int],
        linear_sizes: Sequence[int],
        activation: nn.Module,
    ):
        super().__init__()

        self.linear_layers = nn.ModuleList()
        input_size = input_shape[0]
        for i in range(len(linear_sizes)):
            self.linear_layers.append(nn.Linear(input_size, linear_sizes[i]))
            input_size = linear_sizes[i]

        mid_size = 625
        self.mid_linear_layer = nn.Linear(input_size, mid_size)

        input_size = conv_sizes[0]
        self.conv_layers = nn.ModuleList()
        for i in range(1, len(conv_sizes)):
            self.conv_layers.append(
                UpResidual1DBlock(
                    input_size,
                    conv_sizes[i],
                    kernel_size=4,
                    strides=2,
                    padding=1,
                    activation=activation,
                )
            )
            input_size = conv_sizes[i]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear_layer in self.linear_layers:
            x = linear_layer(x)

        x = self.mid_linear_layer(x)
        x = x.unsqueeze(1)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        return x
