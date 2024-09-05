from typing import Tuple
import torch.nn as nn
import torch
import numpy as np


class DownResidual1DBlock(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        kernel_size: int,
        strides: Tuple[int, int],
        padding: int,
        activation: nn.Module = nn.GELU(),
    ):
        super().__init__()
        self.residual = nn.Conv1d(
            input_size,
            output_size,
            kernel_size,
            strides,
            padding=padding,
            padding_mode="circular",
        )

        self.conv1 = nn.Conv1d(
            input_size,
            input_size * 4,
            kernel_size,
            strides,
            padding=padding,
            padding_mode="circular",
        )
        self.conv2 = nn.Conv1d(input_size * 4, output_size, 1, 1, padding=0)
        self.activation = activation
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = x + residual
        x = torch.swapaxes(x, 1, 2)
        x = self.layer_norm(x)
        x = torch.swapaxes(x, 1, 2)
        return x


class UpResidual1DBlock(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        kernel_size: int,
        strides: Tuple[int, int],
        padding: int,
        activation: nn.Module = nn.GELU(),
    ):
        super().__init__()
        self.residual = nn.ConvTranspose1d(
            input_size,
            output_size,
            kernel_size,
            strides,
            padding=padding,
        )

        self.conv1 = nn.ConvTranspose1d(
            input_size,
            input_size * 4,
            kernel_size,
            strides,
            padding=padding,
        )
        self.conv2 = nn.ConvTranspose1d(input_size * 4, output_size, 1, 1, padding=0)
        self.activation = activation
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = x + residual
        x = torch.swapaxes(x, 1, 2)
        x = self.layer_norm(x)
        x = torch.swapaxes(x, 1, 2)
        return x


# class EvoPositionalEmbedding(nn.Module):
#     """EvoPositionalEmbedding module for Flax."""

#     hidden_size: int
#     max_seq_len: int
#     dtype: jnp.dtype

#     @nn.compact
#     def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
#         x = nn.Dense(self.hidden_size, dtype=self.dtype, name="input_layer")(x)
#         pos = jnp.arange(x.shape[1], dtype=jnp.int16)
#         pos_emb = nn.Embed(
#             num_embeddings=self.max_seq_len,
#             features=self.hidden_size,
#             dtype=self.dtype,
#             name="pos_emb",
#         )(pos)
#         pos_emb = pos_emb.astype(self.dtype)
#         x = x + pos_emb[None, : x.shape[1]]
#         return x


# class TransformerEncoderBlock(nn.Module):
#     """TransformerEncoder module for Flax."""

#     hidden_size: int
#     num_heads: int
#     causal_mask: bool
#     dtype: jnp.dtype
#     dropout_rate: float = 0.05
#     mask: jnp.ndarray | None = None
#     train: bool = True

#     @nn.compact
#     def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

#         attn_outs = nn.MultiHeadAttention(
#             num_heads=self.num_heads,
#             qkv_features=x.shape[-1] * 2,
#             out_features=x.shape[-1],
#             dropout_rate=self.dropout_rate,
#             deterministic=not self.train,
#             dtype=self.dtype,
#             force_fp32_for_softmax=True,
#             # normalize_qk=True,
#         )(x, x, x, mask=self.mask)

#         x = x + nn.Dropout(rate=self.dropout_rate)(
#             attn_outs, deterministic=not self.train
#         )
#         x = nn.LayerNorm(dtype=self.dtype)(x)

#         # MLP block
#         linear_outs = nn.Dense(
#             self.hidden_size * 4, dtype=self.dtype, name="mlp_expand"
#         )(x)
#         linear_outs = nn.gelu(linear_outs)
#         linear_outs = nn.Dense(self.hidden_size, dtype=self.dtype, name="mlp_contract")(
#             linear_outs
#         )
#         x = x + nn.Dropout(rate=self.dropout_rate)(
#             linear_outs, deterministic=not self.train
#         )
#         x = nn.LayerNorm(dtype=self.dtype)(x)

#         return x
