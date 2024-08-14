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
import functools


class EvolvedLatentTransformer(nn.Module):
    """EvolvedLatentTransformer module for Flax."""

    hidden_size: int
    num_heads: int
    num_outputs: int
    max_seq_len: int
    num_layers: int
    causal_mask: bool
    dtype: jnp.dtype

    @nn.compact
    def __call__(
        self, x: jax.Array, mask: jax.Array | None = None, train: bool = True
    ) -> jax.Array:
        if mask is None and self.causal_mask:
            mask = nn.make_causal_mask(x, dtype=jnp.bool_)

        x = nn.Dense(self.hidden_size, dtype=self.dtype, name="input_layer")(x)
        pos = jnp.arange(x.shape[1], dtype=jnp.int16)
        pos_emb = nn.Embed(
            num_embeddings=self.max_seq_len,
            features=self.hidden_size,
            dtype=self.dtype,
            name="pos_emb",
        )(pos)
        pos_emb = pos_emb.astype(self.dtype)
        x = x + pos_emb[None, : x.shape[1]]

        # Transformer blocks.
        block_fn = functools.partial(
            nn.MultiHeadAttention,
            num_heads=self.num_heads,
            qkv_features=x.shape[-1] * 4,
            out_features=x.shape[-1],
            dropout_rate=0.05,
            deterministic=not train,
            dtype=self.dtype,
            force_fp32_for_softmax=True,
            normalize_qk=True,
        )
        block_fn = nn.remat(block_fn, prevent_cse=False)
        block = block_fn(name="block")
        x, _ = nn.scan(
            lambda module, carry, _: (module(carry), None),
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            length=self.num_layers,
        )(block, x, ())

        # Output layer.
        x = nn.LayerNorm(dtype=self.dtype, name="post_norm")(x)
        x = nn.Dense(
            features=self.num_outputs,
            dtype=self.dtype,
            name="output_layer",
        )(x)
        x = x.astype(jnp.float32)
        return x

    @staticmethod
    def create(
        hidden_size: int,
        num_heads: int,
        num_outputs: int,
        max_seq_len: int,
        num_layers: int,
        causal_mask: bool,
        dtype: DType,
    ) -> "EvolvedLatentTransformer":
        if isinstance(dtype, str):
            dtype = _str_to_dtype[dtype]
        return EvolvedLatentTransformer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_outputs=num_outputs,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            causal_mask=causal_mask,
            dtype=dtype,
        )
