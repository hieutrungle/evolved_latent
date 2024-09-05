from typing import Any, Tuple, Callable, Dict
from evolved_latent.trainers.trainer_module import TrainerModule
import numpy as np
import torch
import torch.nn as nn
import os

# Type aliases
PyTree = Any


class Evaluator(TrainerModule):
    pass

    # def __init__(
    #     self,
    #     binded_autoencoder: flax.linen.Module,
    #     binded_evo_model: flax.linen.Module,
    #     **kwargs,
    # ):
    #     super().__init__(**kwargs)
    #     self.binded_autoencoder = binded_autoencoder
    #     self.binded_evo_model = binded_evo_model

    # def create_step_functions(self):

    #     def mse_loss(params, batch, train, rng_key):
    #         x: jnp.ndarray = batch[0]
    #         y: jnp.ndarray = batch[1]
    #         x = self.binded_autoencoder.encoder.apply(
    #             {"params": self.binded_autoencoder.encoder.variables["params"]},
    #             x,
    #             train=train,
    #             rngs={"dropout": rng_key},
    #         )
    #         x = jnp.expand_dims(x, axis=-1)
    #         pred = self.binded_evo_model.apply(
    #             {"params": self.binded_evo_model.variables["params"]}, x, train=train
    #         )
    #         pred = jnp.squeeze(pred, axis=-1)
    #         y_pred = self.binded_autoencoder.decoder.apply(
    #             {"params": self.binded_autoencoder.decoder.variables["params"]},
    #             pred,
    #             train=train,
    #             rngs={"dropout": rng_key},
    #         )
    #         axes = tuple(range(1, len(y.shape)))
    #         loss = jnp.sum(jnp.mean((y_pred - y) ** 2, axis=axes))
    #         return loss

    #     def accumulate_gradients(state, batch, rng_key):
    #         batch_size = batch[0].shape[0]
    #         num_minibatches = self.grad_accum_steps
    #         minibatch_size = batch_size // self.grad_accum_steps
    #         rngs = jax.random.split(rng_key, num_minibatches)
    #         grad_fn = jax.value_and_grad(mse_loss)

    #         def _minibatch_step(
    #             minibatch_idx: jax.Array | int,
    #         ) -> Tuple[PyTree, jnp.ndarray]:
    #             """Determine gradients and metrics for a single minibatch."""
    #             minibatch = jax.tree_map(
    #                 lambda x: jax.lax.dynamic_slice_in_dim(  # Slicing with variable index (jax.Array).
    #                     x,
    #                     start_index=minibatch_idx * minibatch_size,
    #                     slice_size=minibatch_size,
    #                     axis=0,
    #                 ),
    #                 batch,
    #             )
    #             step_loss, step_grads = grad_fn(
    #                 state.params,
    #                 minibatch,
    #                 train=True,
    #                 rng_key=rngs[minibatch_idx],
    #             )
    #             return step_loss, step_grads

    #         def _scan_step(
    #             carry: Tuple[PyTree, jnp.ndarray], minibatch_idx: jax.Array | int
    #         ) -> Tuple[Tuple[PyTree, jnp.ndarray], None]:
    #             """Scan step function for looping over minibatches."""
    #             step_loss, step_grads = _minibatch_step(minibatch_idx)
    #             carry = jax.tree_map(jnp.add, carry, (step_loss, step_grads))
    #             return carry, None

    #         # Determine initial shapes for gradients and loss.
    #         loss_shape, grads_shapes = jax.eval_shape(_minibatch_step, 0)
    #         grads = jax.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), grads_shapes)
    #         loss = jax.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), loss_shape)

    #         # Loop over minibatches to determine gradients and metrics.
    #         (loss, grads), _ = jax.lax.scan(
    #             _scan_step,
    #             init=(loss, grads),
    #             xs=jnp.arange(num_minibatches),
    #             length=num_minibatches,
    #         )

    #         # Average gradients over minibatches.
    #         grads = jax.tree_map(lambda g: g / num_minibatches, grads)
    #         return loss, grads

    #     def train_step(state, batch):
    #         rng, step_rng = jax.random.split(state.rng)
    #         loss, grads = accumulate_gradients(state, batch, step_rng)
    #         state = state.apply_gradients(grads=grads, rng=rng)
    #         metrics = {"loss": loss}
    #         return state, metrics

    #     def eval_step(state, batch):
    #         loss = mse_loss(state.params, batch, train=False, rng_key=state.rng)
    #         return {"loss": loss}

    #     return train_step, eval_step
