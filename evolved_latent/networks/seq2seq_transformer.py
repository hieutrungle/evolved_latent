from typing import Tuple, Sequence, Any
from jax import numpy as jnp
from flax import linen as nn
from evolved_latent.networks.network_utils import (
    Activation,
    _str_to_activation,
    DType,
    _str_to_dtype,
)


class Seq2SeqTransformer(nn.Module):
    """ResNetNormAutoencoder module for Flax."""

    # encoder_config: dict
    # decoder_config: dict

    def setup(self):
        # self.encoder = ResNetNormEncoder.create(**self.encoder_config)
        # self.decoder = ResNetNormDecoder.create(**self.decoder_config)
        pass

    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # x = self.encoder(x)
        # x = self.decoder(x)
        # return x
        pass

    @staticmethod
    def create(
        top_sizes: Sequence[int],
        mid_sizes: Sequence[int],
        bottom_sizes: Sequence[int],
        dense_sizes: Sequence[int],
        activation: Activation,
        dtype: DType = jnp.bfloat16,
    ) -> "Seq2SeqTransformer":
        # encoder_config = {
        #     "top_sizes": top_sizes[1:],
        #     "mid_sizes": mid_sizes[1:],
        #     "bottom_sizes": bottom_sizes[1:],
        #     "dense_sizes": dense_sizes[1:],
        #     "activation": activation,
        #     "dtype": dtype,
        # }
        # decoder_config = {
        #     "top_sizes": top_sizes[:-1][::-1],
        #     "mid_sizes": mid_sizes[:-1][::-1],
        #     "bottom_sizes": bottom_sizes[:-1][::-1],
        #     "dense_sizes": dense_sizes[:-1][::-1],
        #     "activation": activation,
        #     "dtype": dtype,
        # }
        # model = Seq2SeqTransformer(encoder_config, decoder_config)
        # return model
        pass
