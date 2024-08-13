from typing import Tuple, Sequence, Any
from jax import numpy as jnp
from flax import linen as nn
from evolved_latent.networks.network_utils import (
    Activation,
    _str_to_activation,
    DType,
    _str_to_dtype,
)
from evolved_latent.networks.common_blocks import DownResidualBlock, UpResidualBlock


class ResNetNormAutoencoder(nn.Module):
    """ResNetNormAutoencoder module for Flax."""

    encoder_config: dict
    decoder_config: dict

    def setup(self):
        self.encoder = ResNetNormEncoder.create(**self.encoder_config)
        self.decoder = ResNetNormDecoder.create(**self.decoder_config)

    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    @staticmethod
    def create(
        top_sizes: Sequence[int],
        mid_sizes: Sequence[int],
        bottom_sizes: Sequence[int],
        dense_sizes: Sequence[int],
        activation: Activation,
        dtype: DType = jnp.bfloat16,
    ) -> "ResNetNormAutoencoder":
        encoder_config = {
            "top_sizes": top_sizes[1:],
            "mid_sizes": mid_sizes[1:],
            "bottom_sizes": bottom_sizes[1:],
            "dense_sizes": dense_sizes[1:],
            "activation": activation,
            "dtype": dtype,
        }
        decoder_config = {
            "top_sizes": top_sizes[:-1][::-1],
            "mid_sizes": mid_sizes[:-1][::-1],
            "bottom_sizes": bottom_sizes[:-1][::-1],
            "dense_sizes": dense_sizes[:-1][::-1],
            "activation": activation,
            "dtype": dtype,
        }
        model = ResNetNormAutoencoder(encoder_config, decoder_config)
        return model


class ResNetNormEncoder(nn.Module):
    """ResNetNormEncoder module for Flax."""

    top_sizes: Sequence[int]
    mid_sizes: Sequence[int]
    bottom_sizes: Sequence[int]
    dense_sizes: Sequence[int]
    activation: nn.activation
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        for size in self.top_sizes:
            x = nn.Conv(size, (3, 3, 3), (2, 2, 2), padding="SAME", dtype=self.dtype)(x)
            x = self.activation(x)
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        dense_fn = nn.remat(nn.Dense, prevent_cse=True)
        x = dense_fn(x.shape[-1], dtype=self.dtype)(x)

        for size in self.mid_sizes:
            x = DownResidualBlock(
                size,
                (5, 5),
                (2, 2),
                padding="VALID",
                activation=self.activation,
                dtype=self.dtype,
            )(x)
            x = self.activation(x)
            x = nn.GroupNorm(num_groups=1, dtype=self.dtype)(x)

        down_res_block_fn = nn.remat(DownResidualBlock, prevent_cse=True)
        for size in self.bottom_sizes:
            x = down_res_block_fn(
                size,
                (3, 3),
                (2, 2),
                padding="SAME",
                activation=self.activation,
                dtype=self.dtype,
            )(x)
            x = self.activation(x)
            x = nn.GroupNorm(num_groups=1, dtype=self.dtype)(x)

        x = jnp.reshape(x, (x.shape[0], -1))
        for size in self.dense_sizes:
            x = nn.Dense(size, dtype=self.dtype)(x)
            x = self.activation(x)

        return x.astype(jnp.float32)

    @staticmethod
    def create(
        top_sizes: Sequence[int],
        mid_sizes: Sequence[int],
        bottom_sizes: Sequence[int],
        dense_sizes: Sequence[int],
        activation: Activation,
        dtype: DType,
    ) -> "ResNetNormEncoder":
        if isinstance(dtype, str):
            dtype = _str_to_dtype[dtype]
        if isinstance(activation, str):
            activation = _str_to_activation[activation]
        model = ResNetNormEncoder(
            top_sizes, mid_sizes, bottom_sizes, dense_sizes, activation, dtype
        )
        return model


class ResNetNormDecoder(nn.Module):
    """ResNetNormDecoder module for Flax."""

    top_sizes: Sequence[int]
    mid_sizes: Sequence[int]
    bottom_sizes: Sequence[int]
    dense_sizes: Sequence[int]
    activation: nn.activation
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:

        for size in self.dense_sizes:
            x = nn.Dense(size, dtype=self.dtype)(x)
            x = self.activation(x)

        x = jnp.reshape(x, (x.shape[0], 2, 2, -1))

        up_res_block_fn = nn.remat(UpResidualBlock, prevent_cse=True)
        for size in self.bottom_sizes:
            x = up_res_block_fn(
                size,
                (3, 3),
                (2, 2),
                padding="SAME",
                activation=self.activation,
                dtype=self.dtype,
            )(x)
            x = self.activation(x)
            x = nn.GroupNorm(num_groups=1, dtype=self.dtype)(x)

        for size in self.mid_sizes:
            x = UpResidualBlock(
                size,
                (5, 5),
                (2, 2),
                padding="VALID",
                activation=self.activation,
                dtype=self.dtype,
            )(x)
            x = self.activation(x)
            x = nn.GroupNorm(num_groups=1, dtype=self.dtype)(x)

        dense_fn = nn.remat(nn.Dense, prevent_cse=True)
        x = dense_fn(x.shape[-1], dtype=self.dtype)(x)
        x = jnp.reshape(x, (*x.shape[:-1], 50, 2 * len(self.top_sizes)))

        for size in self.top_sizes:
            x = nn.ConvTranspose(
                size, (3, 3, 3), (2, 2, 2), padding="SAME", dtype=self.dtype
            )(x)
            x = self.activation(x)

        return x.astype(jnp.float32)

    @staticmethod
    def create(
        top_sizes: Sequence[int],
        mid_sizes: Sequence[int],
        bottom_sizes: Sequence[int],
        dense_sizes: Sequence[int],
        activation: Activation,
        dtype: DType,
    ) -> "ResNetNormDecoder":
        if isinstance(dtype, str):
            dtype = _str_to_dtype[dtype]
        if isinstance(activation, str):
            activation = _str_to_activation[activation]
        model = ResNetNormDecoder(
            top_sizes, mid_sizes, bottom_sizes, dense_sizes, activation, dtype
        )
        return model
