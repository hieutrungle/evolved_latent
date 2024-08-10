"""Console script for basyesian_optimization_circuit."""

import sys
import numpy as np
import evolved_latent
from evolved_latent.trainer import (
    evolved_autoencoder_trainer,
    baseline_autoencoder_trainer,
)
from evolved_latent.utils import dataloader
import os
import jax
import jax.numpy as jnp
import importlib.resources
import time


def mse_loss_fn(anply_fn, params, x, y_true):
    y_pred = anply_fn(params, x)
    loss = jnp.sum(jnp.mean((y_pred - y_true) ** 2, axis=0))
    return loss


def main():
    print(f"Number of GPUs: {jax.device_count(backend='gpu')}")
    lib_dir = importlib.resources.files(evolved_latent)
    source_dir = os.path.dirname(lib_dir)
    data_dir = os.path.join(source_dir, "local_data", "vel_field_vtk")

    data_shape = (100, 100, 200, 1)
    batch_size = 4
    workers = 4
    train_ds = dataloader.FlameGenerator(
        data_dir,
        batch_size=batch_size,
        data_shape=data_shape,
        workers=workers,
        use_multiprocessing=True,
    )
    eval_ds = dataloader.FlameGenerator(
        data_dir,
        batch_size=batch_size,
        data_shape=data_shape,
        workers=workers,
        use_multiprocessing=True,
        eval_mode=True,
    )

    current_time = time.strftime("%Y%m%d-%H%M%S")
    trainer_config = {
        # "model_class": autoencoder.EvolvedAutoencoder,
        "model_hparams": {
            "top_sizes": (1, 2, 4),
            "mid_sizes": (200, 200, 400),
            "bottom_sizes": (400, 512),
            "dense_sizes": (1024, 256, 64),
            "activation": "relu",
        },
        "optimizer_hparams": {
            "optimizer": "adamw",
            "lr": 1e-3,
        },
        "exmp_input": train_ds[0][0],
        "seed": 0,
        "logger_params": {
            "log_dir": os.path.join(source_dir, "logs"),
            "log_name": os.path.join("evolved_latent_" + current_time),
        },
        "check_val_every_n_epoch": 1,
    }
    num_epochs = 4

    trainer = baseline_autoencoder_trainer.AutoencoderTrainer(**trainer_config)
    # trainer = evolved_autoencoder_trainer.AutoencoderTrainer(**trainer_config)

    eval_metrics = trainer.train_model(
        train_ds, train_ds, train_ds, num_epochs=num_epochs
    )
    print(f"Eval metrics: \n{eval_metrics}")


if __name__ == "__main__":
    main()
