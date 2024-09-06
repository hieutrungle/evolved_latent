from typing import Sequence
import torch
import torch.nn as nn
from evolved_latent.networks.network_utils import Activation, _str_to_activation
from evolved_latent.networks.common_blocks import DownResidual1DBlock, UpResidual1DBlock
from evolved_latent.networks import vector_quantizer
import numpy as np


class VQVAE(nn.Module):
    """AEBaseline module for Pytorch."""

    def __init__(
        self,
        input_shape: Sequence[int],
        conv_sizes: Sequence[int],
        activation: Activation,
    ):
        super().__init__()
        if isinstance(activation, str):
            activation = _str_to_activation[activation]
        encoder_config = {
            "input_shape": input_shape,
            "conv_sizes": conv_sizes,
            "activation": activation,
        }
        self.encoder = VQEncoder(**encoder_config)

        input_shape = self.encoder(torch.zeros(1, *input_shape)).shape[1:]
        decoder_config = {
            "input_shape": input_shape,
            "conv_sizes": conv_sizes[::-1],
            "activation": activation,
        }
        self.decoder = VQDecoder(**decoder_config)

        embedding_dim = conv_sizes[-1]
        self.vq_layer = vector_quantizer.VectorQuantizer1DEMA(256, embedding_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        z_loss, z_quantized, z_perplexity, _ = self.vq_layer(x)
        x_hat = self.decode(z_quantized)
        return x_hat, z_loss, z_perplexity


class VQEncoder(nn.Module):
    """VQEncoder module for Pytorch."""

    def __init__(
        self,
        input_shape: Sequence[int],
        conv_sizes: Sequence[int],
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        return x


class VQDecoder(nn.Module):
    """VQDecoder module for Pytorch."""

    def __init__(
        self,
        input_shape: Sequence[int],
        conv_sizes: Sequence[int],
        activation: nn.Module,
    ):
        super().__init__()

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
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        return x
