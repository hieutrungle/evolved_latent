import jax
from jax import numpy as jnp
from evolved_latent.trainers.trainer_module import TrainerModule


class AutoencoderTrainer(TrainerModule):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_step_functions(self):

        key = jax.random.PRNGKey(self.seed)
        key, dropout_key = jax.random.split(key)

        def mse_loss(params, batch, train):
            x, y = batch
            pred = self.model.apply(
                {"params": params}, x, train=train, rngs={"dropout": dropout_key}
            )
            loss = jnp.sum(jnp.mean((pred - y) ** 2, axis=0))
            return loss

        def train_step(state, batch):
            loss_fn = lambda params: mse_loss(params, batch, train=True)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            metrics = {"loss": loss}
            return state, metrics

        def eval_step(state, batch):
            loss = mse_loss(state.params, batch, train=False)
            return {"loss": loss}

        return train_step, eval_step
