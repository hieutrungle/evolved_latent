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
import orbax.checkpoint as ocp


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

    autoencoder_hparams = {
        "top_sizes": (1, 2, 4),
        "mid_sizes": (200, 200, 400),
        "bottom_sizes": (400, 512),
        "dense_sizes": (1024, 256, 64),
        "activation": "gelu",
        "dtype": "bfloat16",
    }
    if args.autoencoder_type == "baseline":
        autoencoder_class = networks.autoencoder_baseline.BaselineAutoencoder
    elif args.autoencoder_type == "resnet":
        autoencoder_class = networks.autoencoder_resnet.ResNetAutoencoder
    elif args.autoencoder_type == "resnet_norm":
        autoencoder_class = networks.autoencoder_resnet_norm.ResNetNormAutoencoder
    elif args.autoencoder_type == "res_attn":
        autoencoder_class = networks.autoencoder_res_attn.ResNetAttentionAutoencoder
    elif args.autoencoder_type == "res_attn_qk":
        autoencoder_class = (
            networks.autoencoder_res_attn_qk.ResNetAttentionQKAutoencoder
        )
    else:
        raise ValueError(f"Autoencoder type {args.autoencoder_type} not supported.")

    evo_hparams = {
        "hidden_size": args.evo_hidden_size,
        "max_seq_len": 200,
        "num_heads": args.evo_num_heads,
        "num_layers": args.evo_num_layers,
        "num_outputs": 1,
        "causal_mask": False,
        "dtype": "bfloat16",
    }
    if args.evo_type == "transformer":
        evo_class = networks.evolved_latent_transformer.EvolvedLatentTransformer
    else:
        raise ValueError(f"EvolvedLatent type {args.evo_type} not supported.")

    if args.command == "train_autoencoder":
        train_autoencoder(args, autoencoder_class, autoencoder_hparams)

    elif args.command == "train_evo":
        if args.autoencoder_checkpoint is None:
            raise ValueError("Autoencoder checkpoint must be provided.")
        train_evo(args, evo_class, evo_hparams, autoencoder_class, autoencoder_hparams)

    elif args.command == "evolve_latent":
        if args.evo_checkpoint is None or args.autoencoder_checkpoint is None:
            raise ValueError("EvolvedLatent checkpoint must be provided.")
        evolve_latent(
            args, evo_class, evo_hparams, autoencoder_class, autoencoder_hparams
        )

    else:
        raise ValueError(f"Command {args.command} not supported.")


def load_model(model_class, model_hparams, checkpoint, example_input, seed=0):
    model = model_class.create(**model_hparams)
    model_rng = jax.random.PRNGKey(seed)
    model_rng, init_rng = jax.random.split(model_rng)
    variables = model.init(init_rng, example_input, train=True)

    checkpoint = os.path.abspath(checkpoint)
    options = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint,
        options=options,
    )
    step = checkpoint_manager.best_step()
    state_dict = checkpoint_manager.restore(
        step,
        args=ocp.args.StandardRestore(
            {"params": variables["params"], "batch_stats": variables.get("batch_stats")}
        ),
    )
    if "batch_stats" in variables:
        variables["batch_stats"] = state_dict["batch_stats"]
    print(f"Model: {model}")
    return model, variables


def evolve_latent(args, evo_class, evo_hparams, autoencoder_class, autoencoder_hparams):
    lib_dir = importlib.resources.files(evolved_latent)
    source_dir = os.path.dirname(lib_dir)
    data_dir = os.path.join(source_dir, "local_data", "vel_field_vtk")

    data_shape = (100, 100, 200, 1)

    # Load autoencoder
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    exmp_input = jax.random.normal(init_key, (1, *data_shape))
    autoencoder, variables = load_model(
        autoencoder_class,
        autoencoder_hparams,
        args.autoencoder_checkpoint,
        exmp_input,
        args.seed,
    )
    autoencoder = autoencoder.bind(variables)

    # Load EvolvedLatent model
    exmp_input = autoencoder.encoder(exmp_input)
    evo_model, variables = load_model(
        evo_class, evo_hparams, args.evo_checkpoint, exmp_input, args.seed
    )
    evo_model = evo_model.bind(variables)
    del variables

    val_ds = dataloader.SequenceGenerator(
        data_dir=data_dir,
        batch_size=args.batch_size,
        data_shape=data_shape,
        is_train=False,
    )


def train_evo(args, evo_class, evo_hparams, autoencoder_class, autoencoder_hparams):
    lib_dir = importlib.resources.files(evolved_latent)
    source_dir = os.path.dirname(lib_dir)
    data_dir = os.path.join(source_dir, "local_data", "vel_field_vtk")

    data_shape = (100, 100, 200, 1)

    # Load autoencoder
    autoencoder = autoencoder_class.create(**autoencoder_hparams)
    model_rng = jax.random.PRNGKey(args.seed)
    model_rng, init_rng = jax.random.split(model_rng)
    exmp_input = jax.random.normal(init_rng, (1, *data_shape))
    variables = autoencoder.init(init_rng, exmp_input, train=True)

    ae_checkpoint = args.autoencoder_checkpoint
    ae_checkpoint = os.path.abspath(ae_checkpoint)
    options = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
    checkpoint_manager = ocp.CheckpointManager(
        ae_checkpoint,
        options=options,
    )
    step = checkpoint_manager.best_step()
    state_dict = checkpoint_manager.restore(
        step,
        args=ocp.args.StandardRestore(
            {"params": variables["params"], "batch_stats": variables.get("batch_stats")}
        ),
    )
    if "batch_stats" in variables:
        variables["batch_stats"] = state_dict["batch_stats"]
    autoencoder = autoencoder.bind(variables)
    print(f"Autoencoder: {autoencoder}")
    encoder = autoencoder.encoder

    # Data loaders
    train_ds = dataloader.SequenceGenerator(
        data_dir=data_dir,
        batch_size=args.batch_size,
        data_shape=data_shape,
    )

    val_ds = dataloader.SequenceGenerator(
        data_dir=data_dir,
        batch_size=args.batch_size,
        data_shape=data_shape,
        is_train=False,
    )

    train_loader, val_loader = dataloader.create_data_loaders(
        train_ds,
        val_ds,
        train=[True, False],
        batch_size=args.batch_size,
        num_workers=args.workers,
        seed=args.seed,
    )

    # Trainer
    current_time = time.strftime("%Y%m%d-%H%M%S")
    exmp_input = encoder(exmp_input)
    exmp_input = jnp.expand_dims(exmp_input, axis=-1)
    model_hparams = evo_hparams
    trainer_config = {
        "model_class": evo_class,
        "model_hparams": model_hparams,
        "optimizer_hparams": {
            "optimizer": "adamw",
            "lr": 1e-3,
        },
        "exmp_input": exmp_input,
        "grad_accum_steps": args.grad_accum_steps,
        "seed": args.seed,
        "logger_params": {
            "log_dir": os.path.join(source_dir, "logs"),
            "log_name": os.path.join("evolved_latent_" + current_time),
        },
        "check_val_every_n_epoch": 1,
    }

    trainer_config["logger_params"]["log_name"] = (
        trainer_config["model_class"].__name__ + "_" + current_time
    )

    trainer = trainers.evolved_latent_trainer.EvolvedLatentTrainer(
        binded_autoencoder=autoencoder, **trainer_config
    )
    trainer.print_class_variables()

    print(f"*" * 80)
    print(f"Training {trainer.model_class.__name__} model")
    eval_metrics = trainer.train_model(
        train_loader, val_loader, val_loader, num_epochs=args.num_epochs
    )
    print(f"Eval metrics: \n{eval_metrics}")


def train_autoencoder(args, autoencoder_class, autoencoder_hparams):

    lib_dir = importlib.resources.files(evolved_latent)
    source_dir = os.path.dirname(lib_dir)
    data_dir = os.path.join(source_dir, "local_data", "vel_field_vtk")

    data_shape = (100, 100, 200, 1)

    train_ds = dataloader.FlameGenerator(
        data_dir,
        batch_size=args.batch_size,
        data_shape=data_shape,
    )
    val_ds = dataloader.FlameGenerator(
        data_dir,
        batch_size=args.batch_size,
        data_shape=data_shape,
        is_train=False,
    )

    train_loader, val_loader = dataloader.create_data_loaders(
        train_ds,
        val_ds,
        train=[True, False],
        batch_size=args.batch_size,
        num_workers=args.workers,
        seed=args.seed,
    )

    current_time = time.strftime("%Y%m%d-%H%M%S")
    exmp_input = jax.random.normal(jax.random.PRNGKey(args.seed), (1, *data_shape))
    trainer_config = {
        "model_class": autoencoder_class,
        "model_hparams": autoencoder_hparams,
        "optimizer_hparams": {
            "optimizer": "adamw",
            "lr": 1e-3,
        },
        "exmp_input": exmp_input,
        "grad_accum_steps": args.grad_accum_steps,
        "seed": args.seed,
        "logger_params": {
            "log_dir": os.path.join(source_dir, "logs"),
            "log_name": os.path.join("evolved_latent_" + current_time),
        },
        "check_val_every_n_epoch": 1,
    }
    trainer_config["logger_params"]["log_name"] = (
        trainer_config["model_class"].__name__ + "_" + current_time
    )
    trainer = trainers.autoencoder_trainer.AutoencoderTrainer(**trainer_config)
    trainer.print_class_variables()

    print(f"*" * 80)
    print(f"Training {trainer.model_class.__name__} model")
    eval_metrics = trainer.train_model(
        train_loader, val_loader, val_loader, num_epochs=args.num_epochs
    )
    print(f"Eval metrics: \n{eval_metrics}")


def parse_agrs():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config_file", "-dcfg", type=str, required=True)
    parser.add_argument("--command", "-cmd", type=str, required=True)
    parser.add_argument("--autoencoder_type", type=str, required=True, default="resnet")
    parser.add_argument("--evo_type", type=str, default="transformer")
    parser.add_argument("--evo_hidden_size", type=int, default=128)
    parser.add_argument("--evo_num_heads", type=int, default=8)
    parser.add_argument("--evo_num_layers", type=int, default=6)
    parser.add_argument("--autoencoder_checkpoint", type=str, default=None)
    parser.add_argument("--evo_checkpoint", type=str, default=None)
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
