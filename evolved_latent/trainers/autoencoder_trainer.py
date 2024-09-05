from typing import Any, Tuple, Callable, Dict
from evolved_latent.trainers.trainer_module import TrainerModule
import numpy as np
import torch
import torch.nn as nn
import os

# Type aliases
PyTree = Any


class AutoencoderTrainer(TrainerModule):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_model_optimizer(self, num_epochs: int, num_steps_per_epoch: int) -> None:
        """
        Initializes the optimizer for the model's components:
        - actor
        - critics
        - target critics
        - other components
        """
        # Initialize optimizer for actor and critic
        (self.optimizer, self.scheduler) = self.init_optimizer(
            self.model,
            self.optimizer_hparams,
            num_epochs,
            num_steps_per_epoch,
        )

    def init_gradient_scaler(self):
        if "cuda" in self.device.type:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            raise f"Device {self.device.type} not supported."

    def save_models(self, step: int):
        """
        Save the model's parameters to a file.
        """
        ckpt = {
            "step": step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.config,
        }
        torch.save(ckpt, os.path.join(self.logger.log_dir, f"checkpoints.pt"))

    def load_models(self) -> int:
        """
        Load the agent's parameters from a file.
        """
        ckpt = torch.load(os.path.join(self.logger.log_dir, f"checkpoints.pt"))
        self.model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer", None) is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
        step = ckpt.get("step", 0)
        return step

    def create_step_functions(self):

        def mse_loss(params, batch, train, rng_key):
            x, y = batch
            pred = self.model.apply(
                {"params": params}, x, train=train, rngs={"dropout": rng_key}
            )
            axes = tuple(range(1, len(y.shape)))
            loss = torch.sum(torch.mean((pred - y) ** 2, axis=axes))
            return loss

        def train_step(state, batch):
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                loss, grads = accumulate_gradients(state, batch, step_rng)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.critics.parameters(), max_norm=1.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            return info

        def eval_step(state, batch):
            loss = mse_loss(state.params, batch, train=False, rng_key=state.rng)
            return {"loss": loss}

        return train_step, eval_step
