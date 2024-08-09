import jax
from jax import numpy as jnp
from evolved_latent.trainer.trainer_module import TrainerModule
from evolved_latent.networks.autoencoder import EvolvedAutoencoder


class AutoencoderTrainer(TrainerModule):

    def __init__(self, **kwargs):
        super().__init__(model_class=EvolvedAutoencoder, **kwargs)

    def create_step_functions(self):
        def mse_loss(params, batch):
            x, y = batch
            pred = self.model.apply({"params": params}, x)
            loss = jnp.sum(jnp.mean((pred - y) ** 2, axis=0))
            return loss

        def train_step(state, batch):
            loss_fn = lambda params: mse_loss(params, batch)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            metrics = {"loss": loss}
            return state, metrics

        def eval_step(state, batch):
            loss = mse_loss(state.params, batch)
            return {"loss": loss}

        return train_step, eval_step
