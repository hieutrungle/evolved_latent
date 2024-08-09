"""Console script for basyesian_optimization_circuit."""

import sys
import numpy as np
import evolved_latent
from evolved_latent.utils import dataloader, trainer_module
from evolved_latent.networks import autoencoder
import os
import jax
import jax.numpy as jnp
import importlib.resources


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
    batch_size = 16
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

    seed = 0
    key = jax.random.PRNGKey(seed)
    top_sizes = (1, 2, 4)
    mid_sizes = (200, 200, 400)
    bottom_sizes = (400, 512)
    dense_sizes = (1024, 256, 64)
    model = autoencoder.EvolvedAutoencoder.create(
        key,
        top_sizes=top_sizes,
        mid_sizes=mid_sizes,
        bottom_sizes=bottom_sizes,
        dense_sizes=dense_sizes,
        activation="relu",
    )

    input_shape = data_shape
    num_epochs = 250
    lr = 1e-3
    num_train_steps = num_epochs * len(train_ds)
    checkpoint_path = os.path.join(source_dir, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)
    trainer = trainer_module.TrainerModule(
        model,
        input_shape=input_shape,
        lr=lr,
        num_train_steps=num_train_steps,
        checkpoint_path=checkpoint_path,
        loss_fn=mse_loss_fn,
        seed=seed,
    )

    trainer.train_model(num_epochs, train_ds, eval_ds)


if __name__ == "__main__":
    main()
