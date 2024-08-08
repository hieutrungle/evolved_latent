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


class EvolvedNet(nn.Module):
    """EvolvedNet module for Flax."""

    @struct.dataclass
    class Config:
        """Config dataclass for EvolvedNet."""

        num_layers: int = 3
        num_units: int = 128
        activation: Activation = "relu"

    config: Config

    def setup(self):
        layers = [
            nn.Dense(self.config.num_units),
            _str_to_activation[self.config.activation],
        ]

        for _ in range(self.config.num_layers - 1):
            layers.extend(
                [
                    nn.Dense(self.config.num_units),
                    _str_to_activation[self.config.activation],
                ]
            )
        self.layers = layers

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x

    @staticmethod
    def create(key: KeyArray, config: Config) -> "EvolvedNet":
        model = EvolvedNet(config)
        return model


class EvolvedAutoencoder(nn.Module):
    """EvolvedAutoencoder module for Flax."""

    # @struct.dataclass
    # class Config:
    #     """Config dataclass for EvolvedAutoencoder."""

    #     encoder: EvolvedEncoder.Config = EvolvedEncoder.Config()
    #     decoder: EvolvedDecoder.Config = EvolvedDecoder.Config()

    key: jax.random.PRNGKey
    encoder_config: dict
    decoder_config: dict

    # config: Config

    def setup(self):
        self.encoder = EvolvedEncoder.create(self.key, **self.encoder_config)
        self.decoder = EvolvedDecoder.create(self.key, **self.decoder_config)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    @staticmethod
    def create(
        key: jax.random.PRNGKey,
        top_sizes: Sequence[int],
        mid_sizes: Sequence[int],
        bottom_sizes: Sequence[int],
        dense_sizes: Sequence[int],
        activation: Activation,
    ) -> "EvolvedAutoencoder":
        encoder_config = {
            "top_sizes": top_sizes[1:],
            "mid_sizes": mid_sizes[1:],
            "bottom_sizes": bottom_sizes[1:],
            "dense_sizes": dense_sizes[1:],
            "activation": activation,
        }
        decoder_config = {
            "top_sizes": top_sizes[:-1][::-1],
            "mid_sizes": mid_sizes[:-1][::-1],
            "bottom_sizes": bottom_sizes[:-1][::-1],
            "dense_sizes": dense_sizes[:-1][::-1],
            "activation": activation,
        }
        model = EvolvedAutoencoder(key, encoder_config, decoder_config)
        return model


class EvolvedEncoder(nn.Module):
    """EvolvedEncoder module for Flax."""

    # @struct.dataclass
    # class Config:
    #     """Config dataclass for EvolvedEncoder."""

    #     first_layer_size: int = 4
    #     mid_sizes: Sequence[int] = (128, 256, 512)
    #     activation: Activation = "relu"

    # config: Config

    top_sizes: Sequence[int]
    mid_sizes: Sequence[int]
    bottom_sizes: Sequence[int]
    dense_sizes: Sequence[int]
    activation: Activation = "relu"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for size in self.top_sizes:
            x = nn.Conv(size, (3, 3, 3), (2, 2, 2), padding="SAME")(x)
            x = _str_to_activation[self.activation](x)
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        # TODO: use attention instead of dense layers
        x = nn.Dense(x.shape[-1])(x)

        for size in self.mid_sizes:
            x = nn.Conv(size, (5, 5), (2, 2), padding="VALID")(x)
            x = _str_to_activation[self.activation](x)

        for size in self.bottom_sizes:
            x = nn.Conv(size, (3, 3), (2, 2), padding="SAME")(x)
            x = _str_to_activation[self.activation](x)

        x = jnp.reshape(x, (x.shape[0], -1))
        for size in self.dense_sizes:
            x = nn.Dense(size)(x)
            x = _str_to_activation[self.activation](x)
        return x

    @staticmethod
    def create(
        key: KeyArray,
        top_sizes: Sequence[int],
        mid_sizes: Sequence[int],
        bottom_sizes: Sequence[int],
        dense_sizes: Sequence[int],
        activation: Activation,
    ) -> "EvolvedEncoder":
        model = EvolvedEncoder(
            top_sizes, mid_sizes, bottom_sizes, dense_sizes, activation
        )
        return model


class EvolvedDecoder(nn.Module):
    """EvolvedDecoder module for Flax."""

    top_sizes: Sequence[int]
    mid_sizes: Sequence[int]
    bottom_sizes: Sequence[int]
    dense_sizes: Sequence[int]
    activation: Activation = "relu"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        for size in self.dense_sizes:
            x = nn.Dense(size)(x)
            x = _str_to_activation[self.activation](x)

        x = jnp.reshape(x, (x.shape[0], 2, 2, -1))

        for size in self.bottom_sizes:
            x = nn.ConvTranspose(size, (3, 3), (2, 2), padding="SAME")(x)
            x = _str_to_activation[self.activation](x)

        for size in self.mid_sizes:
            x = nn.ConvTranspose(size, (5, 5), (2, 2), padding="VALID")(x)
            x = _str_to_activation[self.activation](x)

        x = nn.Dense(x.shape[-1])(x)
        x = jnp.reshape(x, (*x.shape[:-1], 50, 2 * len(self.top_sizes)))

        for size in self.top_sizes:
            x = nn.ConvTranspose(size, (3, 3, 3), (2, 2, 2), padding="SAME")(x)
            x = _str_to_activation[self.activation](x)

        return x

    @staticmethod
    def create(
        key: KeyArray,
        top_sizes: Sequence[int],
        mid_sizes: Sequence[int],
        bottom_sizes: Sequence[int],
        dense_sizes: Sequence[int],
        activation: Activation,
    ) -> "EvolvedDecoder":
        model = EvolvedDecoder(
            top_sizes, mid_sizes, bottom_sizes, dense_sizes, activation
        )
        return model


def encode(key: KeyArray, x: jnp.ndarray) -> jnp.ndarray:
    enc_model = EvolvedEncoder.create(
        key,
        top_sizes=(2, 4),
        mid_sizes=(256, 512),
        bottom_sizes=(1024,),
        dense_sizes=(1024, 256, 64),
        activation="relu",
    )
    model_tx = optax.adam(1e-3)
    enc_state = train_state.TrainState.create(
        apply_fn=enc_model.apply, params=enc_model.init(key, x), tx=model_tx
    )
    tabulate_fn = nn.tabulate(
        enc_model, key, compute_flops=True, compute_vjp_flops=True
    )
    print(tabulate_fn(x))
    return enc_state.apply_fn(enc_state.params, x)


def decode(key: KeyArray, x: jnp.ndarray) -> jnp.ndarray:
    dec_model = EvolvedDecoder.create(
        key,
        top_sizes=(512,),
        mid_sizes=(256, 200),
        bottom_sizes=(2, 1),
        dense_sizes=(256, 1024, 4096),
        activation="relu",
    )
    model_tx = optax.adam(1e-3)
    dec_state = train_state.TrainState.create(
        apply_fn=dec_model.apply, params=dec_model.init(key, x), tx=model_tx
    )
    tabulate_fn = nn.tabulate(
        dec_model, key, compute_flops=True, compute_vjp_flops=True
    )
    print(tabulate_fn(x))
    return dec_state.apply_fn(dec_state.params, x)


@jax.jit
def train_step(
    key: jax.random.PRNGKey, state: train_state.TrainState, x: jnp.ndarray
) -> train_state.TrainState:
    print(f"Compilation!")

    def loss_fn(params: nn.Module, x: jnp.ndarray) -> jnp.ndarray:
        x_hat = state.apply_fn(params, x)
        return jnp.mean((x - x_hat) ** 2)

    loss, grad = jax.value_and_grad(loss_fn)(state.params, x)
    state = state.apply_gradients(grads=grad)

    return state


def train_autoencoder(
    key: jax.random.PRNGKey, state: train_state.TrainState, ds: jnp.ndarray
) -> train_state.TrainState:
    for i, x in enumerate(ds):
        print(f"Batch {i}")
        start_time = time.time()
        state = train_step(key, state, x)
        print(f"Time taken: {time.time() - start_time}")
    return state


def create_autoencoder(
    key: jax.random.PRNGKey, input_shape: Tuple[int]
) -> train_state.TrainState:
    x_tmp = jax.random.normal(key, input_shape)
    # Add a batch dimension and a channel dimension
    x_tmp = jnp.expand_dims(x_tmp, axis=0)
    x_tmp = jnp.expand_dims(x_tmp, axis=-1)

    top_sizes = (1, 2, 4)
    mid_sizes = (200, 200, 400)
    bottom_sizes = (400, 512)
    dense_sizes = (1024, 256, 64)
    model = EvolvedAutoencoder.create(
        key,
        top_sizes=top_sizes,
        mid_sizes=mid_sizes,
        bottom_sizes=bottom_sizes,
        dense_sizes=dense_sizes,
        activation="relu",
    )
    model_tx = optax.adam(1e-3)
    autoencoder_state = train_state.TrainState.create(
        apply_fn=model.apply, params=model.init(key, x_tmp), tx=model_tx
    )
    tabulate_fn = nn.tabulate(model, key, compute_flops=True, compute_vjp_flops=True)
    print(tabulate_fn(x_tmp))
    return autoencoder_state


if __name__ == "__main__":
    # Example of using the Evolved
    # Create a model
    key = random.PRNGKey(0)
    input_shape = (100, 100, 200)
    model_state = create_autoencoder(key, input_shape)

    # create fake dataset
    num_batches = 10
    batch_size = 2
    ds = jax.random.normal(key, (num_batches, batch_size, *input_shape))
    ds = jnp.expand_dims(ds, axis=-1)
    print(f"Dataset shape: {ds.shape}")

    # Train the model
    model_state = train_autoencoder(key, model_state, ds)
    print("Training done")

    y = model_state.apply_fn(model_state.params, ds[0])
    print(f"AUTOENCODER:: Input shape: {ds[0].shape}, Output shape: {y.shape}")
