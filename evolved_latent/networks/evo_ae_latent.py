from typing import Tuple, Sequence, Any
import jax
from jax import numpy as jnp
from flax import linen as nn
from evolved_latent.networks.network_utils import (
    Activation,
    _str_to_activation,
    DType,
    _str_to_dtype,
)
from evolved_latent.networks.common_blocks import (
    EvoPositionalEmbedding,
    TransformerEncoderBlock,
)
import functools


class EvoAutoencoder(nn.Module):
    """EvolvedLatentTransformer module for Flax."""

    encoder: nn.Module
    evolver: nn.Module
    decoder: nn.Module

    @nn.compact
    def __call__(
        self, x: jax.Array, mask: jax.Array | None = None, train: bool = True
    ) -> jax.Array:
        x = self.encoder(x, train=train)
        x = jnp.expand_dims(x, axis=-1)
        x = self.evolver(x, mask=mask, train=train)
        x = jnp.squeeze(x, axis=-1)
        x = self.decoder(x, train=train)
        return x

    @staticmethod
    def create(
        binded_autoencoder: nn.Module,
        binded_evolver: nn.Module,
        dtype: DType,
    ) -> "EvoAutoencoder":
        if isinstance(dtype, str):
            dtype = _str_to_dtype[dtype]
        encoder, _ = binded_autoencoder.encoder.unbind()
        decoder, _ = binded_autoencoder.decoder.unbind()
        evolver, _ = binded_evolver.unbind()
        return EvoAutoencoder(
            encoder=encoder,
            evolver=evolver,
            decoder=decoder,
        )
