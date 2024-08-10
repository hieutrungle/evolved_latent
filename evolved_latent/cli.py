"""Console script for evolved_latent."""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import importlib.resources
import argparse
import evolved_latent
from evolved_latent.trainer import (
    evolved_autoencoder_trainer,
    baseline_autoencoder_trainer,
)
from evolved_latent.utils import dataloader
import jax
import jax.numpy as jnp
import time


def main():
    args = parse_agrs()
    print(f"Number of GPUs: {jax.device_count(backend='gpu')}")

    lib_dir = importlib.resources.files(evolved_latent)
    source_dir = os.path.dirname(lib_dir)
    data_dir = os.path.join(source_dir, "local_data", "vel_field_vtk")

    data_shape = (100, 100, 200, 1)
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    workers = args.workers
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

    trainer = baseline_autoencoder_trainer.AutoencoderTrainer(**trainer_config)
    # trainer = evolved_autoencoder_trainer.AutoencoderTrainer(**trainer_config)

    eval_metrics = trainer.train_model(
        train_ds, train_ds, train_ds, num_epochs=num_epochs
    )
    print(f"Eval metrics: \n{eval_metrics}")


def parse_agrs():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config_file", "-dcfg", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
