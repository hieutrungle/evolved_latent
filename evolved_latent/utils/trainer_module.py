import jax
from jax import numpy as jnp
from flax import linen as nn  # nn notation also used in PyTorch and in Flax's older API
from flax.training import train_state
import orbax.checkpoint as ocp
import optax
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter

import os
import glob
from typing import Callable, Optional, Tuple, Sequence, Union, Any


class TrainerModule:

    def __init__(
        self,
        model: nn.Module,
        input_shape: Sequence[int],
        lr: float,
        num_train_steps: int,
        checkpoint_path: str,
        loss_fn: Callable,
        seed: int = 0,
    ):
        super().__init__()

        self.key = jax.random.PRNGKey(seed)
        optimizer = create_optimizer(lr, num_train_steps)
        x_tmp = jax.random.normal(self.key, (1, *input_shape))
        self.model_state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=model.init(self.key, x_tmp),
            tx=optimizer,
        )

        self.checkpoint_path = checkpoint_path
        if self.checkpoint_path != "":
            self.checkpoint_path = os.path.abspath(self.checkpoint_path)
            options = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
            self.checkpoint_manager = ocp.CheckpointManager(
                self.checkpoint_path,
                options=options,
            )

        self.logger = SummaryWriter(log_dir=self.checkpoint_path)

        def train_step(
            state: train_state.TrainState, x: jnp.ndarray, y: jnp.ndarray
        ) -> Tuple[train_state.TrainState, jnp.ndarray]:
            jax_loss_fn = lambda params: loss_fn(state.apply_fn, params, x, y)
            loss, grad = jax.value_and_grad(jax_loss_fn)(state.params)
            state = state.apply_gradients(grads=grad)
            return state, loss

        def eval_step(
            state: train_state.TrainState, x: jnp.ndarray, y: jnp.ndarray
        ) -> jnp.ndarray:
            return loss_fn(state.apply_fn, state.params, x, y)

        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)

    def train_model(self, num_epochs, train_loader, val_loader):
        # Train model for defined number of epochs
        best_eval = np.inf
        for epoch_idx in tqdm(range(1, num_epochs + 1)):
            self.train_epoch(epoch=epoch_idx, train_loader=train_loader)
            if epoch_idx % 10 == 0:
                eval_loss = self.eval_model(val_loader)
                self.logger.add_scalar("val/loss", eval_loss, global_step=epoch_idx)
                if eval_loss < best_eval:
                    best_eval = eval_loss
                    self.save_model(step=epoch_idx)
                self.logger.flush()
        self.wait_for_checkpoint()

    def train_epoch(self, epoch: int, train_loader):
        # Train model for one epoch, and log avg loss
        losses = []
        for batch in train_loader:
            (x, y) = batch
            self.model_state, loss = self.train_step(self.model_state, x, y)
            losses.append(loss)
        losses_np = np.stack(jax.device_get(losses))
        avg_loss = losses_np.mean()
        self.logger.add_scalar("train/loss", avg_loss, global_step=epoch)

    def eval_model(self, data_loader):
        # Test model on all images of a data loader and return avg loss
        losses = []
        batch_sizes = []
        for batch in data_loader:
            (x, y) = batch
            loss = self.eval_step(self.model_state, x, y)
            losses.append(loss)
            batch_sizes.append(batch[0].shape[0])
        losses_np = np.stack(jax.device_get(losses))
        batch_sizes_np = np.stack(batch_sizes)
        avg_loss = (losses_np * batch_sizes_np).sum() / batch_sizes_np.sum()
        return avg_loss

    def save_model(self, step: int):
        """
        Save the agent's parameters to a file.
        """
        self.checkpoint_manager.save(
            step, args=ocp.args.StandardSave(self.model_state.params)
        )

    def wait_for_checkpoint(self):
        """
        Wait for the checkpoint manager to finish writing checkpoints.
        """
        self.checkpoint_manager.wait_until_finished()

    def load_model(self, step: int = None):
        """
        Load the agent's parameters from a file.
        """
        if step == None:
            step = self.checkpoint_manager.best_step()
        params = self.checkpoint_manager.restore(
            step, args=ocp.args.StandardRestore(self)
        )
        self.model_state = train_state.TrainState.create(
            apply_fn=self.model_state.apply_fn,
            params=params,
            tx=self.model_state.tx,
        )

    def is_checkpoint_exists(self):
        # Check whether a pretrained model exist for this autoencoder
        folders = glob.glob(os.path.join(self.checkpoint_path, "*"))
        numeric_folders = [folder for folder in folders if folder.isdigit()]
        return len(numeric_folders) > 0


@optax.inject_hyperparams
def chain_optimizer(learning_rate: float):
    return optax.chain(optax.clip(1.0), optax.adamw(learning_rate=learning_rate))


def create_optimizer(
    learning_rate: float, num_train_steps: int
) -> optax.GradientTransformation:
    init_value = learning_rate / 50
    end_value = learning_rate / 5
    warmup_steps = num_train_steps // 5
    decay_steps = num_train_steps
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=init_value,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=end_value,
    )
    optimizer = chain_optimizer(learning_rate=schedule)
    return optimizer
