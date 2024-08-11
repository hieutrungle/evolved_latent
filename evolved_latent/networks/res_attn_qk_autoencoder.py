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


class ResNetAttentionQKAutoencoder(nn.Module):
    """ResNetAttentionQKAutoencoder module for Flax."""

    encoder_config: dict
    decoder_config: dict

    def setup(self):
        self.encoder = ResNetAttentionQKEncoder.create(**self.encoder_config)
        self.decoder = ResNetAttentionQKDecoder.create(**self.decoder_config)

    def __call__(
        self, x: jnp.ndarray, train: bool = True, dropout_rng=None
    ) -> jnp.ndarray:
        x = self.encoder(x, train=train, dropout_rng=dropout_rng)
        x = self.decoder(x, train=train, dropout_rng=dropout_rng)
        return x

    @staticmethod
    def create(
        top_sizes: Sequence[int],
        mid_sizes: Sequence[int],
        bottom_sizes: Sequence[int],
        dense_sizes: Sequence[int],
        activation: Activation,
        dtype: DType = jnp.bfloat16,
    ) -> "ResNetAttentionQKAutoencoder":
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
        model = ResNetAttentionQKAutoencoder(encoder_config, decoder_config)
        return model


class ResNetAttentionQKEncoder(nn.Module):
    """ResNetAttentionQKEncoder module for Flax."""

    top_sizes: Sequence[int]
    mid_sizes: Sequence[int]
    bottom_sizes: Sequence[int]
    dense_sizes: Sequence[int]
    activation: nn.activation
    dtype: jnp.dtype

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool = False, dropout_rng=None
    ) -> jnp.ndarray:
        for size in self.top_sizes:
            x = nn.Conv(size, (3, 3, 3), (2, 2, 2), padding="SAME", dtype=self.dtype)(x)
            x = self.activation(x)
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        q = x
        k = nn.Dense(x.shape[-1], dtype=self.dtype)(x)
        x = nn.MultiHeadAttention(
            num_heads=10,
            qkv_features=x.shape[-1],
            out_features=x.shape[-1],
            dropout_rate=0.1,
            deterministic=not train,
            dtype=self.dtype,
        )(q, k, dropout_rng=dropout_rng)

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

        for size in self.bottom_sizes:
            x = DownResidualBlock(
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
    ) -> "ResNetAttentionQKEncoder":
        if isinstance(dtype, str):
            dtype = _str_to_dtype[dtype]
        if isinstance(activation, str):
            activation = _str_to_activation[activation]
        model = ResNetAttentionQKEncoder(
            top_sizes, mid_sizes, bottom_sizes, dense_sizes, activation, dtype
        )
        return model


class ResNetAttentionQKDecoder(nn.Module):
    """ResNetAttentionQKDecoder module for Flax."""

    # key: jax.random.PRNGKey
    top_sizes: Sequence[int]
    mid_sizes: Sequence[int]
    bottom_sizes: Sequence[int]
    dense_sizes: Sequence[int]
    activation: nn.activation
    dtype: jnp.dtype

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool = False, dropout_rng=None
    ) -> jnp.ndarray:

        for size in self.dense_sizes:
            x = nn.Dense(size, dtype=self.dtype)(x)
            x = self.activation(x)

        x = jnp.reshape(x, (x.shape[0], 2, 2, -1))

        for size in self.bottom_sizes:
            x = UpResidualBlock(
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

        q = x
        k = nn.Dense(x.shape[-1], dtype=self.dtype)(x)
        x = nn.MultiHeadAttention(
            num_heads=10,
            qkv_features=x.shape[-1],
            out_features=x.shape[-1],
            dropout_rate=0.1,
            deterministic=not train,
            dtype=self.dtype,
        )(q, k, dropout_rng=dropout_rng)

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
    ) -> "ResNetAttentionQKDecoder":
        if isinstance(dtype, str):
            dtype = _str_to_dtype[dtype]
        if isinstance(activation, str):
            activation = _str_to_activation[activation]
        model = ResNetAttentionQKDecoder(
            top_sizes, mid_sizes, bottom_sizes, dense_sizes, activation, dtype
        )
        return model
