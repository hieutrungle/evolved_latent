import os
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union
import json
import time
from tqdm.auto import tqdm
import numpy as np
import copy
from evolved_latent.utils.logger import TensorboardLogger
import argparse
from evolved_latent.utils import utils, pytorch_utils
import re
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchinfo import summary
from torch import optim
import glob


class TrainerModule:

    def __init__(
        self,
        model_class: nn.Module,
        model_hparams: Dict[str, Any],
        optimizer_hparams: Dict[str, Any],
        input_shape: Sequence[int],
        grad_accum_steps: int = 1,
        seed: int = 42,
        logger_params: Dict[str, Any] = None,
        enable_progress_bar: bool = True,
        debug: bool = False,
        check_val_every_n_epoch: int = 10,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        """
        A basic Trainer module summarizing most common training functionalities
        like logging, model initialization, training loop, etc.

        Atributes:
          model_class: The class of the model that should be trained.
          model_hparams: A dictionary of all hyperparameters of the model. Is
            used as input to the model when created.
          optimizer_hparams: A dictionary of all hyperparameters of the optimizer.
            Used during initialization of the optimizer.
          exmp_input: Input to the model for initialization and tabulate.
          seed: Seed to initialize PRNG.
          logger_params: A dictionary containing the specification of the logger.
          enable_progress_bar: If False, no progress bar is shown.
          debug: If True, no jitting is applied. Can be helpful for debugging.
          check_val_every_n_epoch: The frequency with which the model is evaluated
            on the validation set.
        """
        super().__init__()
        self.model_class = model_class
        self.model_hparams = model_hparams
        self.optimizer_hparams = optimizer_hparams
        self.enable_progress_bar = enable_progress_bar
        self.debug = debug
        self.seed = seed
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.grad_accum_steps = grad_accum_steps
        self.dtype = dtype

        # Set of hyperparameters to save
        self.config = {
            "model_class": model_class.__name__,
            "model_hparams": model_hparams,
            "optimizer_hparams": optimizer_hparams,
            "logger_params": logger_params,
            "enable_progress_bar": self.enable_progress_bar,
            "debug": self.debug,
            "grad_accum_steps": grad_accum_steps,
            "check_val_every_n_epoch": check_val_every_n_epoch,
            "seed": self.seed,
        }
        self.config.update(kwargs)

        # Create model
        self.model = self.create_model(self.model_class, self.model_hparams)
        batch_input_shape = [1, *input_shape]
        self.summarize_model(self.model, [batch_input_shape])

        # Init trainer parts
        self.logger = self.init_logger(logger_params)
        (self.train_step, self.eval_step, self.update_step) = (
            self.create_step_functions()
        )

    def create_model(self, model_class: Callable, model_hparams: Dict[str, Any]):
        """
        Create a model from a given class and hyperparameters.

        Args:
            model_class: The class of the model that should be created.
            model_hparams: A dictionary of all hyperparameters of the model.
              Is used as input to the model when created.

        Returns:
            model: The created model.
        """
        create_fn = getattr(model_class, "create", None)
        model: nn.Module = None
        if callable(create_fn):
            model = create_fn(**model_hparams)
        else:
            model = model_class(**model_hparams)
        return model

    def summarize_model(self, model: nn.Module, input_shapes: Sequence[Sequence[int]]):
        """
        Prints a summary of the Module represented as table.

        Args:
          input_shapes: A list of input shapes to the model.
        """
        print(f"Model: {model.__class__.__name__}")
        summary(model, input_size=[*input_shapes])
        print()

    def init_logger(self, logger_params: Optional[Dict] = None):
        """
        Initializes a logger and creates a logging directory.

        Args:
          logger_params: A dictionary containing the specification of the logger.
        """
        if logger_params is None:
            logger_params = dict()
        # Determine logging directory
        log_dir = logger_params.get("log_dir", None)
        if not log_dir:
            base_log_dir = logger_params.get("base_log_dir", "checkpoints/")
            # Prepare logging
            log_dir = os.path.join(base_log_dir, self.config["model_class"])
            version = None
        else:
            version = ""

        # Create logger object
        if "log_name" in logger_params:
            log_dir = os.path.join(log_dir, logger_params["log_name"])
        logger_type = logger_params.get("logger_type", "TensorBoard").lower()
        if logger_type == "tensorboard":
            logger = TensorboardLogger(log_dir=log_dir, comment=version)
        elif logger_type == "wandb":
            logger = WandbLogger(
                name=logger_params.get("project_name", None),
                save_dir=log_dir,
                version=version,
                config=self.config,
            )
        else:
            assert False, f'Unknown logger type "{logger_type}"'

        # Save config hyperparameters
        log_dir = logger.log_dir
        if not os.path.isfile(os.path.join(log_dir, "hparams.json")):
            os.makedirs(os.path.join(log_dir, "metrics/"), exist_ok=True)
            with open(os.path.join(log_dir, "hparams.json"), "w") as f:
                json.dump(self.config, f, indent=4)

        return logger

    @staticmethod
    def create_step_functions(
        self,
    ) -> Tuple[
        Callable[[nn.Module, Dict[str, torch.Tensor]], Dict[str, Any]],
        Callable[[nn.Module, Dict[str, torch.Tensor]], Dict[str, Any]],
        Callable[[nn.Module, Dict[str, torch.Tensor]], Dict[str, Any]],
    ]:
        """
        Creates and returns functions for the training and evaluation step. The
        functions take as input the training state and a batch from the train/
        val/test loader. Both functions are expected to return a dictionary of
        logging metrics, and the training function a new train state. This
        function needs to be overwritten by a subclass. The train_step and
        eval_step functions here are examples for the signature of the functions.
        """

        def train_step(agent: nn.Module, batch: Any):
            metrics = {}
            return metrics

        def eval_step(agent: nn.Module, batch: Any):
            metrics = {}
            return metrics

        def update_step(agent: nn.Module, batch: Any):
            metrics = {}
            return metrics

        raise NotImplementedError

    def init_optimizer(
        self,
        module: nn.Module,
        optimizer_hparams: Dict[str, Any],
        num_epochs: int,
        num_steps_per_epoch: int,
    ) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """
        Initializes the optimizer and learning rate scheduler.

        Args:
            module: The module for which the optimizer should be initialized.
            optimizer_hparams: A dictionary of all hyperparameters of the optimizer.
            num_epochs: Number of epochs the model will be trained for.
            num_steps_per_epoch: Number of training steps per epoch.
        """
        hparams = copy.copy(optimizer_hparams)

        # Initialize optimizer
        optimizer_name = hparams.pop("optimizer", "adamw")
        if optimizer_name.lower() == "adam":
            opt_class = optim.Adam
        elif optimizer_name.lower() == "adamw":
            opt_class = optim.AdamW
        elif optimizer_name.lower() == "sgd":
            opt_class = optim.SGD
        else:
            assert False, f'Unknown optimizer "{opt_class}"'

        # Initialize learning rate scheduler
        # A cosine decay scheduler is used
        lr = hparams.pop("lr", 1e-3)
        num_train_steps = int(num_epochs * num_steps_per_epoch)
        warmup_steps = hparams.pop("warmup_steps", num_train_steps // 5)

        optimizer = opt_class(module.parameters(), lr=lr, **hparams)
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1 / 20, total_iters=warmup_steps
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, num_train_steps - warmup_steps, eta_min=lr / 10
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_scheduler, cosine_scheduler], [warmup_steps]
        )

        return optimizer, scheduler

    @staticmethod
    def init_model_optimizer(self, num_epochs: int, num_steps_per_epoch: int) -> None:
        """
        Initializes the optimizer for the agent's components:
        - actor
        - critics
        - target critics
        - other components

        Returns:
            agent: The agent with initialized optimizers.
        """
        raise NotImplementedError

    @staticmethod
    def init_gradient_scaler(self):
        """
        Initializes the gradient scaler for mixed precision training.
        """
        raise NotImplementedError

    def train_model(
        self,
        train_loader: Iterator,
        val_loader: Iterator,
        test_loader: Optional[Iterator] = None,
        num_epochs: int = 500,
        args: argparse.Namespace = None,
    ) -> Dict[str, Any]:
        """
        Starts a training loop for the given number of epochs.

        Args:
          train_loader: Data loader of the training set.
          val_loader: Data loader of the validation set.
          test_loader: If given, best model will be evaluated on the test set.
          num_epochs: Number of epochs for which to train the model.

        Returns:
          A dictionary of the train, validation and evt. test metrics for the
          best model on the validation set.
        """
        assert num_epochs > 0, "Number of epochs must be larger than 0"
        assert (
            num_epochs // self.check_val_every_n_epoch > 0
        ), f"Number of epochs ({num_epochs}) must be larger than check_val_every_n_epoch ({self.check_val_every_n_epoch})"

        print("\n" + f"*" * 80)
        print(f"Training {self.model_class.__name__} and {self.critic_class.__name__}")

        # Create optimizer and the scheduler for the given number of epochs
        self.init_model_optimizer(num_epochs, len(train_loader))

        # Load model if exists
        is_ckpt_available = False
        if args.resume and self.logger.log_dir is not None:
            ckpt_file = glob.glob(os.path.join(self.logger.log_dir, "*.pt"))
            if ckpt_file:
                is_ckpt_available = True
        if is_ckpt_available:
            print(f"Loading model from {self.logger.log_dir}")
            start_step = self.load_models()
            start_step += 1
        else:
            print(f"Training from scratch")
            start_step = 0
        start_step = int(start_step)

        self.init_gradient_scaler()

        # Prepare training loop
        self.on_training_start()
        best_eval_metrics = None
        eval_metrics = {
            "val/loss": 0.0,
        }
        train_metrics = {"train/loss": 0.0}
        t = tqdm(
            range(start_step, num_epochs),
            total=num_epochs,
            dynamic_ncols=True,
            initial=start_step,
        )
        for epoch_idx in t:
            train_metrics = self.train_epoch(train_loader, log_prefix="train/")

            self.logger.log_metrics(train_metrics, step=epoch_idx)
            self.on_training_epoch_end(epoch_idx)

            # Validation every N epochs
            if epoch_idx % self.check_val_every_n_epoch == 0:
                eval_metrics = self.eval_model(val_loader, log_prefix="val/")
                self.on_validation_epoch_end(epoch_idx, eval_metrics, val_loader)
                self.logger.log_metrics(eval_metrics, step=epoch_idx)
                # self.logger.log_metrics(eval_metrics, step=epoch_idx)
                self.save_metrics(f"eval_epoch_{str(epoch_idx).zfill(3)}", eval_metrics)
                # Save best model
                if self.is_new_model_better(eval_metrics, best_eval_metrics):
                    best_eval_metrics = eval_metrics
                    best_eval_metrics.update(train_metrics)
                    self.save_model(step=epoch_idx)
                    self.save_metrics("best_eval", eval_metrics)

            t.set_postfix(
                {
                    "train_loss": f"{train_metrics['train/loss']:.4e}",
                    "val_loss": f"{eval_metrics['val/loss']:.4e}",
                },
                refresh=True,
            )

        # Test best model if possible
        if test_loader is not None:
            self.load_models()
            test_metrics = self.eval_model(test_loader, log_prefix="test/")
            self.logger.log_metrics(test_metrics, step=epoch_idx)
            self.save_metrics("test", test_metrics)
            best_eval_metrics.update(test_metrics)

        # Close logger
        # self.logger.finalize("success")
        return best_eval_metrics

    def train_epoch(
        self, train_loader: Iterator, log_prefix: Optional[str] = "train/"
    ) -> Dict[str, Any]:
        """
        Trains a model for one epoch.

        Args:
          train_loader: Data loader of the training set.
          log_prefix: Prefix to add to all metrics (e.g. 'train/')

        Returns:
          A dictionary of the average training metrics over all batches
          for logging.
        """
        # Train model for one epoch, and log avg loss and accuracy
        metrics = defaultdict(float)
        num_train_steps = len(train_loader)
        start_time = time.time()
        for i, batch in enumerate(train_loader):
            # for batch in self.tracker(train_loader, desc="Training", leave=False):
            self.state, step_metrics = self.train_step(self.state, batch)
            for key in step_metrics:
                metrics[log_prefix + key] += step_metrics[key] / num_train_steps
        metrics = {key: metrics[key].item() for key in metrics}
        metrics["epoch_time"] = time.time() - start_time
        metrics["learning_rate"] = float(
            self.state.opt_state.hyperparams["learning_rate"]
        )
        return metrics

    def eval_model(
        self, data_loader: Iterator, log_prefix: Optional[str] = ""
    ) -> Dict[str, Any]:
        """
        Evaluates the model on a dataset.

        Args:
          data_loader: Data loader of the dataset to evaluate on.
          log_prefix: Prefix to add to all metrics (e.g. 'val/' or 'test/')

        Returns:
          A dictionary of the evaluation metrics, averaged over data points
          in the dataset.
        """
        # Test model on all images of a data loader and return avg loss
        metrics = defaultdict(float)
        num_elements = 0
        for batch in data_loader:
            step_metrics = self.eval_step(self.state, batch)
            batch_size = (
                batch[0].shape[0]
                if isinstance(batch, (list, tuple))
                else batch.shape[0]
            )
            for key, value in step_metrics.items():
                metrics[key] += value * batch_size
            num_elements += batch_size
        metrics = {
            (log_prefix + key): (metrics[key] / num_elements).item() for key in metrics
        }
        return metrics

    def is_new_model_better(
        self, new_metrics: Dict[str, Any], old_metrics: Dict[str, Any]
    ) -> bool:
        """
        Compares two sets of evaluation metrics to decide whether the
        new model is better than the previous ones or not.

        Args:
          new_metrics: A dictionary of the evaluation metrics of the new model.
          old_metrics: A dictionary of the evaluation metrics of the previously
            best model, i.e. the one to compare to.

        Returns:
          True if the new model is better than the old one, and False otherwise.
        """
        if old_metrics is None:
            return True
        for key, is_larger in [
            ("val/val_metric", False),
            ("val/acc", True),
            ("val/loss", False),
        ]:
            if key in new_metrics:
                if is_larger:
                    return new_metrics[key] > old_metrics[key]
                else:
                    return new_metrics[key] < old_metrics[key]
        assert False, f"No known metrics to log on: {new_metrics}"

    def tracker(self, iterator: Iterator, **kwargs) -> Iterator:
        """
        Wraps an iterator in a progress bar tracker (tqdm) if the progress bar
        is enabled.

        Args:
          iterator: Iterator to wrap in tqdm.
          kwargs: Additional arguments to tqdm.

        Returns:
          Wrapped iterator if progress bar is enabled, otherwise same iterator
          as input.
        """
        if self.enable_progress_bar:
            return tqdm(iterator, **kwargs)
        else:
            return iterator

    def save_metrics(self, filename: str, metrics: Dict[str, Any]):
        """
        Saves a dictionary of metrics to file. Can be used as a textual
        representation of the validation performance for checking in the terminal.

        Args:
          filename: Name of the metrics file without folders and postfix.
          metrics: A dictionary of metrics to save in the file.
        """
        with open(os.path.join(self.log_dir, f"metrics/{filename}.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    def on_training_start(self):
        """
        Method called before training is started. Can be used for additional
        initialization operations etc.
        """
        pass

    def on_training_epoch_end(self, epoch_idx: int):
        """
        Method called at the end of each training epoch. Can be used for additional
        logging or similar.

        Args:
          epoch_idx: Index of the training epoch that has finished.
        """
        pass

    def on_validation_epoch_end(
        self, epoch_idx: int, eval_metrics: Dict[str, Any], val_loader: Iterator
    ):
        """
        Method called at the end of each validation epoch. Can be used for additional
        logging and evaluation.

        Args:
          epoch_idx: Index of the training epoch at which validation was performed.
          eval_metrics: A dictionary of the validation metrics. New metrics added to
            this dictionary will be logged as well.
          val_loader: Data loader of the validation set, to support additional
            evaluation.
        """
        pass

    @staticmethod
    def save_models(self, step: int):
        """
        Save the agent's parameters to a file.
        """
        ckpt = {
            "step": step,
            "agent": self.agent.state_dict(),
            "config": self.config,
        }
        torch.save(ckpt, os.path.join(self.logger.log_dir, f"checkpoints.pt"))

    @staticmethod
    def load_models(self) -> int:
        """
        Load the agent's parameters from a file.
        """
        ckpt = torch.load(os.path.join(self.logger.log_dir, f"checkpoints.pt"))
        self.agent.load_state_dict(ckpt["agent"])
        step = ckpt.get("step", 0)
        return step

    @classmethod
    def load_from_checkpoint(cls, checkpoint: str, exmp_input: Any) -> Any:
        """
        Creates a Trainer object with same hyperparameters and loaded model from
        a checkpoint directory.

        Args:
          checkpoint: Folder in which the checkpoint and hyperparameter file is stored.
          exmp_input: An input to the model for shape inference.

        Returns:
          A Trainer object with model loaded from the checkpoint folder.
        """
        hparams_file = os.path.join(checkpoint, "hparams.json")
        assert os.path.isfile(hparams_file), "Could not find hparams file"
        with open(hparams_file, "r") as f:
            hparams = json.load(f)
        hparams.pop("model_class")
        hparams.update(hparams.pop("model_hparams"))
        if not hparams["logger_params"]:
            hparams["logger_params"] = dict()
        hparams["logger_params"]["log_dir"] = checkpoint
        trainer = cls(exmp_input=exmp_input, **hparams)
        trainer.load_model()
        return trainer

    def finalize(self):
        """
        Method called at the end of the training. Can be used for final logging
        or similar.
        """
        pass

    def print_class_variables(self):
        """
        Prints all class variables of the TrainerModule.
        """
        print()
        print(f"*" * 80)
        print(f"Class variables of {self.__class__.__name__}:")
        skipped_keys = ["state", "variables", "encoder", "decoder"]

        def check_for_skipped_keys(key):
            for skip_key in skipped_keys:
                if str(skip_key).lower() in str(key).lower():
                    return True
            return False

        for key, value in self.__dict__.items():
            if not check_for_skipped_keys(key):
                print(f" - {key}: {value}")
        print(f"*" * 80)
        print()
