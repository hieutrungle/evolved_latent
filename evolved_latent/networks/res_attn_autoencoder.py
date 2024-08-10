from typing import Callable, Optional, Tuple, Sequence, Union, Any
import os
from jax._src.typing import Array
from jax._src import dtypes

import numpy as np

import jax
from jax import lax, random, numpy as jnp
from flax import struct
from jax._src import core
from flax import linen as nn  # nn notation also used in PyTorch and in Flax's older API

from flax.training import train_state

# JAX optimizers - a separate lib developed by DeepMind
import optax
import functools
from dataclasses import dataclass

import time

KeyArray = Array
DTypeLikeFloat = Any
DTypeLikeComplex = Any
DTypeLikeInexact = Any  # DTypeLikeFloat | DTypeLikeComplex


Activation = Union[str, Callable]


class Identity(nn.Module):
    """Identity module for Flax."""

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


_str_to_activation = {
    "relu": nn.activation.relu,
    "tanh": nn.activation.tanh,
    "sigmoid": nn.activation.sigmoid,
    "swish": nn.activation.hard_swish,
    "gelu": nn.activation.gelu,
    "elu": nn.activation.elu,
    "identity": Identity(),
}


class ResNetAttentionAutoencoder(nn.Module):
    """ResNetAttentionAutoencoder module for Flax."""

    encoder_config: dict
    decoder_config: dict

    def setup(self):
        self.encoder = ResNetAttentionEncoder.create(**self.encoder_config)
        self.decoder = ResNetAttentionDecoder.create(**self.decoder_config)

    def __call__(
        self, x: jnp.ndarray, train: bool = True, dropout_rng=None
    ) -> jnp.ndarray:
        x = self.encoder(x, train=train, dropout_rng=dropout_rng)
        x = self.decoder(x, train=train, dropout_rng=dropout_rng)
        return x

    @staticmethod
    def create(
        # key: jax.random.PRNGKey,
        top_sizes: Sequence[int],
        mid_sizes: Sequence[int],
        bottom_sizes: Sequence[int],
        dense_sizes: Sequence[int],
        activation: Activation,
    ) -> "ResNetAttentionAutoencoder":
        # enc_key, dec_key = random.split(key)
        encoder_config = {
            # "key": enc_key,
            "top_sizes": top_sizes[1:],
            "mid_sizes": mid_sizes[1:],
            "bottom_sizes": bottom_sizes[1:],
            "dense_sizes": dense_sizes[1:],
            "activation": activation,
        }
        decoder_config = {
            # "key": dec_key,
            "top_sizes": top_sizes[:-1][::-1],
            "mid_sizes": mid_sizes[:-1][::-1],
            "bottom_sizes": bottom_sizes[:-1][::-1],
            "dense_sizes": dense_sizes[:-1][::-1],
            "activation": activation,
        }
        model = ResNetAttentionAutoencoder(encoder_config, decoder_config)
        return model


class ResNetAttentionEncoder(nn.Module):
    """ResNetAttentionEncoder module for Flax."""

    # key: jax.random.PRNGKey
    top_sizes: Sequence[int]
    mid_sizes: Sequence[int]
    bottom_sizes: Sequence[int]
    dense_sizes: Sequence[int]
    activation: Activation = "gelu"

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool = False, dropout_rng=None
    ) -> jnp.ndarray:
        for size in self.top_sizes:
            x = nn.Conv(size, (3, 3, 3), (2, 2, 2), padding="SAME")(x)
            x = _str_to_activation[self.activation](x)
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        q = x
        x = nn.MultiHeadAttention(
            num_heads=10,
            qkv_features=x.shape[-1],
            out_features=x.shape[-1],
            dropout_rate=0.1,
            deterministic=not train,
        )(q, dropout_rng=dropout_rng)

        for size in self.mid_sizes:
            # x = nn.Conv(size, (5, 5), (2, 2), padding="VALID")(x)
            x = DownResidualBlock(
                size, (5, 5), (2, 2), self.activation, padding="VALID"
            )(x)
            x = _str_to_activation[self.activation](x)

        for size in self.bottom_sizes:
            # x = nn.Conv(size, (3, 3), (2, 2), padding="SAME")(x)
            x = DownResidualBlock(
                size, (3, 3), (2, 2), self.activation, padding="SAME"
            )(x)
            x = _str_to_activation[self.activation](x)

        x = jnp.reshape(x, (x.shape[0], -1))
        for size in self.dense_sizes:
            x = nn.Dense(size)(x)
            x = _str_to_activation[self.activation](x)
        return x

    @staticmethod
    def create(
        top_sizes: Sequence[int],
        mid_sizes: Sequence[int],
        bottom_sizes: Sequence[int],
        dense_sizes: Sequence[int],
        activation: Activation,
    ) -> "ResNetAttentionEncoder":
        model = ResNetAttentionEncoder(
            top_sizes, mid_sizes, bottom_sizes, dense_sizes, activation
        )
        return model


class ResNetAttentionDecoder(nn.Module):
    """ResNetAttentionDecoder module for Flax."""

    # key: jax.random.PRNGKey
    top_sizes: Sequence[int]
    mid_sizes: Sequence[int]
    bottom_sizes: Sequence[int]
    dense_sizes: Sequence[int]
    activation: Activation = "gelu"

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool = False, dropout_rng=None
    ) -> jnp.ndarray:

        for size in self.dense_sizes:
            x = nn.Dense(size)(x)
            x = _str_to_activation[self.activation](x)

        x = jnp.reshape(x, (x.shape[0], 2, 2, -1))

        for size in self.bottom_sizes:
            x = UpResidualBlock(size, (3, 3), (2, 2), self.activation, padding="SAME")(
                x
            )
            x = _str_to_activation[self.activation](x)

        for size in self.mid_sizes:
            x = UpResidualBlock(size, (5, 5), (2, 2), self.activation, padding="VALID")(
                x
            )
            x = _str_to_activation[self.activation](x)

        # TODO: use attention instead of dense layers
        q = x
        x = nn.MultiHeadAttention(
            num_heads=10,
            qkv_features=x.shape[-1],
            out_features=x.shape[-1],
            dropout_rate=0.1,
            deterministic=not train,
        )(q, dropout_rng=dropout_rng)
        x = jnp.reshape(x, (*x.shape[:-1], 50, 2 * len(self.top_sizes)))

        for size in self.top_sizes:
            x = nn.ConvTranspose(size, (3, 3, 3), (2, 2, 2), padding="SAME")(x)
            x = _str_to_activation[self.activation](x)

        return x

    @staticmethod
    def create(
        top_sizes: Sequence[int],
        mid_sizes: Sequence[int],
        bottom_sizes: Sequence[int],
        dense_sizes: Sequence[int],
        activation: Activation,
    ) -> "ResNetAttentionDecoder":
        model = ResNetAttentionDecoder(
            top_sizes, mid_sizes, bottom_sizes, dense_sizes, activation
        )
        return model


class DownResidualBlock(nn.Module):
    """ResidualBlock module for Flax."""

    features: int
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int]
    activation: Activation = "gelu"
    padding: str = "SAME"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = nn.Conv(
            self.features, self.kernel_size, self.strides, padding=self.padding
        )(x)
        x = nn.Conv(
            self.features, self.kernel_size, self.strides, padding=self.padding
        )(x)
        x = _str_to_activation[self.activation](x)
        x = nn.Conv(self.features * 4, (1, 1), (1, 1), padding="SAME")(x)
        x = _str_to_activation[self.activation](x)
        x = nn.Conv(self.features, (1, 1), (1, 1), padding="SAME")(x)
        x = x + residual
        return x


class UpResidualBlock(nn.Module):
    """UpResidualBlock module for Flax."""

    features: int
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int]
    activation: Activation = "gelu"
    padding: str = "SAME"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = nn.ConvTranspose(
            self.features, self.kernel_size, self.strides, padding=self.padding
        )(x)
        x = nn.ConvTranspose(
            self.features, self.kernel_size, self.strides, padding=self.padding
        )(x)
        x = _str_to_activation[self.activation](x)
        x = nn.ConvTranspose(self.features * 4, (1, 1), (1, 1), padding="SAME")(x)
        x = _str_to_activation[self.activation](x)
        x = nn.ConvTranspose(self.features, (1, 1), (1, 1), padding="SAME")(x)
        x = x + residual
        return x
