from typing import Sequence, Any

from jax import numpy as jnp
from flax import linen as nn

from evolved_latent.networks.network_utils import (
    Activation,
    _str_to_activation,
    DType,
    _str_to_dtype,
)


class BaselineAutoencoder(nn.Module):
    """BaselineAutoencoder module for Flax."""

    encoder_config: dict
    decoder_config: dict

    def setup(self):
        self.encoder = BaselineEncoder.create(**self.encoder_config)
        self.decoder = BaselineDecoder.create(**self.decoder_config)

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
    ) -> "BaselineAutoencoder":
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
        model = BaselineAutoencoder(encoder_config, decoder_config)
        return model


class BaselineEncoder(nn.Module):
    """BaselineEncoder module for Flax."""

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

        x = nn.Dense(x.shape[-1], dtype=self.dtype)(x)

        for size in self.mid_sizes:
            x = nn.Conv(size, (5, 5), (2, 2), padding="VALID", dtype=self.dtype)(x)
            x = self.activation(x)

        for size in self.bottom_sizes:
            x = nn.Conv(size, (3, 3), (2, 2), padding="SAME", dtype=self.dtype)(x)
            x = self.activation(x)

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
    ) -> "BaselineEncoder":
        if isinstance(dtype, str):
            dtype = _str_to_dtype[dtype]
        if isinstance(activation, str):
            activation = _str_to_activation[activation]
        model = BaselineEncoder(
            top_sizes, mid_sizes, bottom_sizes, dense_sizes, activation, dtype
        )
        return model


class BaselineDecoder(nn.Module):
    """BaselineDecoder module for Flax."""

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

        for size in self.bottom_sizes:
            x = nn.ConvTranspose(
                size, (3, 3), (2, 2), padding="SAME", dtype=self.dtype
            )(x)
            x = self.activation(x)

        for size in self.mid_sizes:
            x = nn.ConvTranspose(
                size, (5, 5), (2, 2), padding="VALID", dtype=self.dtype
            )(x)
            x = self.activation(x)

        x = nn.Dense(x.shape[-1], dtype=self.dtype)(x)
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
    ) -> "BaselineDecoder":
        if isinstance(dtype, str):
            dtype = _str_to_dtype[dtype]
        if isinstance(activation, str):
            activation = _str_to_activation[activation]
        model = BaselineDecoder(
            top_sizes, mid_sizes, bottom_sizes, dense_sizes, activation, dtype
        )
        return model
