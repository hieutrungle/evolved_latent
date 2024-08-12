"""Console script for evolved_latent."""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Jax acceleration flags
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true "
    "--xla_gpu_triton_gemm_any=True "
    # "--xla_gpu_enable_async_collectives=true "
    # "--xla_gpu_enable_latency_hiding_scheduler=true "
    # "--xla_gpu_enable_highest_priority_async_stream=true "
)

import importlib.resources
import argparse
import evolved_latent

from evolved_latent import trainers, networks
from evolved_latent.utils import dataloader
import jax
import jax.numpy as jnp
import time


def main():
    args = parse_agrs()
    print(f"Number of GPUs: {jax.device_count(backend='gpu')}")
    print("Device:", jax.devices()[0])

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
        shuffle=True,
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
            "activation": "gelu",
            "dtype": "bfloat16",
        },
        "optimizer_hparams": {
            "optimizer": "adamw",
            "lr": 1e-3,
        },
        "exmp_input": train_ds[0][0],
        "grad_accum_steps": args.grad_accum_steps,
        "seed": args.seed,
        "logger_params": {
            "log_dir": os.path.join(source_dir, "logs"),
            "log_name": os.path.join("evolved_latent_" + current_time),
        },
        "check_val_every_n_epoch": 1,
    }

    if args.model_type == "baseline":
        trainer_config["model_class"] = (
            networks.baseline_autoencoder.BaselineAutoencoder
        )

    elif args.model_type == "resnet":
        trainer_config["model_class"] = networks.resnet_autoencoder.ResNetAutoencoder

    elif args.model_type == "resnet_norm":
        trainer_config["model_class"] = (
            networks.resnet_norm_autoencoder.ResNetNormAutoencoder
        )

    elif args.model_type == "res_attn":
        trainer_config["model_class"] = (
            networks.res_attn_autoencoder.ResNetAttentionAutoencoder
        )

    elif args.model_type == "res_attn_qk":
        trainer_config["model_class"] = (
            networks.res_attn_qk_autoencoder.ResNetAttentionQKAutoencoder
        )

    else:
        raise ValueError(f"Model type {args.model_type} not supported.")
    trainer_config["logger_params"]["log_name"] = (
        trainer_config["model_class"].__name__ + "_" + current_time
    )
    trainer = trainers.autoencoder_trainer.AutoencoderTrainer(**trainer_config)
    trainer.print_class_variables()

    print(f"*" * 80)
    print(f"Training {trainer_config['model_class'].__name__} model")
    eval_metrics = trainer.train_model(
        train_ds, eval_ds, eval_ds, num_epochs=num_epochs
    )
    print(f"Eval metrics: \n{eval_metrics}")


def parse_agrs():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config_file", "-dcfg", type=str, required=True)
    parser.add_argument(
        "--model_type", "-mt", type=str, required=True, default="resnet"
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
