import jax


class GenerateCallback:

    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs  # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def log_generations(self, model, state, logger, epoch):
        if epoch % self.every_n_epochs == 0:
            reconst_imgs = model.apply({"params": state.params}, self.input_imgs)
            reconst_imgs = jax.device_get(reconst_imgs)

            # Plot and add to tensorboard
            imgs = np.stack([self.input_imgs, reconst_imgs], axis=1).reshape(
                -1, *self.input_imgs.shape[1:]
            )
            imgs = jax_to_torch(imgs)
            grid = torchvision.utils.make_grid(
                imgs, nrow=2, normalize=True, value_range=(-1, 1)
            )
            logger.add_image("Reconstructions", grid, global_step=epoch)


class EarlyStoppingCallback:

    def __init__(self, patience=10):
        super().__init__()
        self.patience = patience
        self.best_eval = 1e6
        self.wait = 0

    def check_early_stopping(self, eval_loss, model, epoch):
        if eval_loss < self.best_eval:
            self.best_eval = eval_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                model.wait_for_checkpoint()
                raise StopIteration
