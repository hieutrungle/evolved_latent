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


class EvoPositionalEmbedding(nn.Module):
    """EvoPositionalEmbedding module for Pytorch."""

    def __init__(
        self, input_size: int, hidden_size: int, max_seq_len: int, dtype: torch.dtype
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.linear = nn.Linear(input_size, hidden_size)
        self.embedding = nn.Embedding(max_seq_len, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        pos = torch.arange(x.shape[1], dtype=torch.int16)
        pos_emb = self.embedding(pos)
        pos_emb = pos_emb.to(self.dtype)
        x = x + pos_emb[None, : x.shape[1]]
        return x


class TransformerENcoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        causal_mask: bool,
        dtype: torch.dtype,
        dropout_rate: float = 0.05,
        mask: torch.Tensor = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.causal_mask = causal_mask
        self.dtype = dtype
        self.dropout_rate = dropout_rate
        self.mask = mask

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_outs, _ = self.attn(
            x, x, x, key_padding_mask=self.mask, need_weights=False
        )
        x = x + self.dropout(attn_outs)
        x = self.layer_norm1(x)

        mlp_outs = self.mlp(x)
        x = x + self.dropout(mlp_outs)
        x = self.layer_norm2(x)

        return x
