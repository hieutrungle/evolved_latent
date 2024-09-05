from typing import Any, Tuple, Callable, Dict
from evolved_latent.trainers.trainer_module import TrainerModule
import numpy as np
import torch
import torch.nn as nn
import os

# Type aliases
PyTree = Any


class VQVAETrainer(TrainerModule):

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
        self.scaler = torch.amp.GradScaler(self.device.type)

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

        def compute_loss(batch):
            x, y = batch
            means, log_stds = self.model.encode(x)
            z = self.model.reparametrize(means, log_stds)
            pred = self.model.decode(z)
            axes = tuple(range(1, len(y.shape)))
            mse_loss = torch.sum(torch.mean((pred - y) ** 2, axis=axes))

            kl_loss = -0.5 * (1 + log_stds - means**2 - torch.exp(log_stds))
            kl_loss = torch.mean(torch.sum(kl_loss, axis=axes))

            loss = mse_loss + 0.75 * kl_loss

            return loss

        def train_step(batch):
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                loss = compute_loss(batch)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            return {"loss": loss}

        def eval_step(batch):
            loss = compute_loss(batch)
            return {"loss": loss}

        return train_step, eval_step
