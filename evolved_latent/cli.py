"""Console script for evolved_latent."""

import os

# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import importlib.resources
import argparse
import evolved_latent
from evolved_latent import trainers, networks
from evolved_latent.utils import dataloader, pytorch_utils, utils
import time
from torchinfo import summary
import torch


def main():
    args = parse_agrs()

    autoencoder_hparams = {
        "input_shape": (1, 20000),
        "conv_sizes": (1, 16, 32, 64, 32, 1),
        "linear_sizes": (512, 64, 32),
        "activation": "gelu",
    }
    if args.autoencoder_type == "baseline":
        autoencoder_class = networks.autoencoder_1d_baseline.AEBaseline
    elif args.autoencoder_type == "vae":
        autoencoder_class = networks.autoencoder_1d_vae.VariationalAutoencoder
    elif args.autoencoder_type == "vqvae":
        autoencoder_hparams = {
            "input_shape": (1, 20000),
            "conv_sizes": (1, 16, 16, 32, 64, 128, 256),
            "activation": "gelu",
            "num_embeddings": 256,
        }
        autoencoder_class = networks.autoencoder_1d_vqvae.VQVAE
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
    # if args.evo_type == "transformer":
    #     evo_class = networks.evolved_latent_transformer.EvolvedLatentTransformer
    # else:
    #     raise ValueError(f"EvolvedLatent type {args.evo_type} not supported.")

    if args.command == "train_autoencoder":
        train_autoencoder(args, autoencoder_class, autoencoder_hparams)

    # elif args.command == "train_evo":
    #     if args.autoencoder_checkpoint is None:
    #         raise ValueError("Autoencoder checkpoint must be provided.")
    #     train_evo(args, evo_class, evo_hparams, autoencoder_class, autoencoder_hparams)

    # elif args.command == "evolve_latent":
    #     if args.evo_checkpoint is None or args.autoencoder_checkpoint is None:
    #         raise ValueError("EvolvedLatent checkpoint must be provided.")
    #     evolve_latent(
    #         args, evo_class, evo_hparams, autoencoder_class, autoencoder_hparams
    #     )

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
    exmp_input = jnp.expand_dims(exmp_input, axis=-1)
    evo_model, variables = load_model(
        evo_class, evo_hparams, args.evo_checkpoint, exmp_input, args.seed
    )
    evo_model = evo_model.bind(variables)
    del variables

    # Combine models
    key, rng_key = jax.random.split(key)
    exmp_input = jax.random.normal(init_key, (1, *data_shape))
    combined_model = networks.evo_ae_latent.EvoAutoencoder.create(
        binded_autoencoder=autoencoder, binded_evolver=evo_model, dtype="bfloat16"
    )
    print(f"Combined model: {combined_model.__class__.__name__}")
    print(combined_model.tabulate(init_key, exmp_input, train=True))

    variables = combined_model.init(init_key, exmp_input, train=True)
    variables["params"]["encoder"] = autoencoder.encoder.variables["params"]
    variables["params"]["decoder"] = autoencoder.decoder.variables["params"]
    variables["params"]["evolver"] = evo_model.variables["params"]

    val_ds = dataloader.SequenceGenerator(
        data_dir=data_dir,
        batch_size=args.batch_size,
        data_shape=data_shape,
        is_train=False,
    )
    [val_loader] = dataloader.create_data_loaders(
        val_ds,
        train=[False],
        batch_size=args.batch_size,
        num_workers=args.workers,
        seed=args.seed,
    )

    def predict(x):
        y_pred = combined_model.apply(
            {"params": variables["params"]},
            x,
            train=False,
            rngs={"dropout": rng_key},
        )
        return y_pred

    def calculate_mse(y_true, y_pred):
        axes = tuple(range(1, len(y_true.shape)))
        loss = jnp.sum(jnp.mean((y_pred - y_true) ** 2, axis=axes))
        return loss

    predict_fn = jax.jit(predict)
    calculate_mse_fn = jax.jit(calculate_mse)

    for i, (inputs, outputs) in enumerate(val_loader):
        num_inputs = inputs.shape[0]
        for j in range(num_inputs):
            x = inputs[j : j + 1]
            y = outputs[j : j + 1]
            y_pred = predict_fn(x)
            mse = calculate_mse_fn(val_ds.denormalize(y), val_ds.denormalize(y_pred))
            # mse = calculate_mse_fn(y, y_pred)
            print(f"Batch {i}, Input {j}, MSE: {mse}")
        print()


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

    input_shape = (1, 20000)

    train_ds = dataloader.FlameGenerator(
        args.data_dir,
        batch_size=args.batch_size,
        data_shape=input_shape,
    )

    val_ds = dataloader.FlameGenerator(
        args.data_dir,
        batch_size=args.batch_size,
        data_shape=input_shape,
        is_train=False,
    )

    print(f"Train DS: {len(train_ds)}")
    print(f"Val DS: {len(val_ds)}")

    train_loader, val_loader = dataloader.create_data_loaders(
        train_ds,
        val_ds,
        train=[True, False],
        batch_size=args.batch_size,
        num_workers=args.workers,
        seed=args.seed,
    )

    current_time = time.strftime("%Y%m%d-%H%M%S")

    trainer_config = {
        "model_class": autoencoder_class,
        "model_hparams": autoencoder_hparams,
        "optimizer_hparams": {
            "optimizer": "adamw",
            "lr": 1e-3,
        },
        "input_shape": input_shape,
        "grad_accum_steps": args.grad_accum_steps,
        "seed": args.seed,
        "logger_params": {
            "log_dir": os.path.join(args.source_dir, "logs"),
            "log_name": os.path.join("evolved_latent_" + current_time),
        },
        "check_val_every_n_epoch": 1,
        "device": args.device,
    }
    trainer_config["logger_params"]["log_name"] = (
        trainer_config["model_class"].__name__ + "_" + current_time
    )
    trainer = trainers.autoencoder_trainer.AutoencoderTrainer(**trainer_config)
    trainer.print_class_variables()

    print(f"*" * 80)
    print(f"Training {trainer.model_class.__name__} model")
    eval_metrics = trainer.train_model(
        train_loader, val_loader, val_loader, num_epochs=args.num_epochs, args=args
    )
    print(f"Eval metrics: \n{eval_metrics}")


def parse_agrs():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config_file", "-dcfg", type=str, required=True)
    parser.add_argument("--command", "-cmd", type=str, required=True)
    parser.add_argument("--autoencoder_type", type=str, default="baseline")
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
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    lib_dir = importlib.resources.files(evolved_latent)
    source_dir = os.path.dirname(lib_dir)
    args.source_dir = source_dir

    device = pytorch_utils.init_gpu()
    args.device = device
    return args


if __name__ == "__main__":
    main()
