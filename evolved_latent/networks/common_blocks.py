from typing import Tuple
from jax import numpy as jnp
from flax import linen as nn


class DownResidualBlock(nn.Module):
    """ResidualBlock module for Flax."""

    features: int
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int]
    padding: str = "SAME"
    activation: nn.activation = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = nn.Conv(
            self.features,
            self.kernel_size,
            self.strides,
            padding=self.padding,
            dtype=self.dtype,
        )(x)
        x = nn.Conv(
            self.features,
            self.kernel_size,
            self.strides,
            padding=self.padding,
            dtype=self.dtype,
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            self.features * 4, (1, 1), (1, 1), padding="SAME", dtype=self.dtype
        )(x)
        x = self.activation(x)
        x = nn.Conv(self.features, (1, 1), (1, 1), padding="SAME", dtype=self.dtype)(x)
        x = x + residual
        return x


class UpResidualBlock(nn.Module):
    """UpResidualBlock module for Flax."""

    features: int
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int]
    padding: str = "SAME"
    activation: nn.activation = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = nn.ConvTranspose(
            self.features,
            self.kernel_size,
            self.strides,
            padding=self.padding,
            dtype=self.dtype,
        )(x)
        x = nn.ConvTranspose(
            self.features,
            self.kernel_size,
            self.strides,
            padding=self.padding,
            dtype=self.dtype,
        )(x)
        x = self.activation(x)
        x = nn.ConvTranspose(
            self.features * 4, (1, 1), (1, 1), padding="SAME", dtype=self.dtype
        )(x)
        x = self.activation(x)
        x = nn.ConvTranspose(
            self.features, (1, 1), (1, 1), padding="SAME", dtype=self.dtype
        )(x)
        x = x + residual
        return x
