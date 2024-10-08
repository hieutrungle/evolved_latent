from typing import Sequence, Union
import torch
import torch.utils.data
import pyvista as pv
import numpy as np
import glob
import os
import flax.linen as nn
import jax

######################################################################
# Tensorflow Data Generator
######################################################################


# class FlameGenerator(tf.keras.utils.PyDataset):
#     def __init__(
#         self,
#         data_dir,
#         batch_size,
#         data_shape,
#         shuffle=False,
#         workers=4,
#         use_multiprocessing=True,
#         max_queue_size=10,
#         eval_mode=False,
#         seed=0,
#         **kwargs,
#     ):
#         super().__init__(
#             workers=workers,
#             use_multiprocessing=use_multiprocessing,
#             max_queue_size=max_queue_size,
#             **kwargs,
#         )
#         self.data_dir = data_dir
#         self.batch_size = batch_size
#         self.data_shape = data_shape
#         self.shuffle = shuffle
#         self.np_rng = np.random.default_rng(seed)

#         self.filenames = glob.glob(data_dir + "/*.vtk", recursive=True)
#         self.filenames.sort(
#             key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1])
#         )
#         # self.filenames = self.filenames[:88]
#         self.num_files = len(self.filenames)

#         if not eval_mode:
#             self.num_files = int(self.num_files * 0.9)
#             self.filenames = self.filenames[: self.num_files]
#         else:
#             self.num_files = int(self.num_files * 0.1)
#             self.filenames = self.filenames[self.num_files :]

#         self.indexes = np.arange(self.num_files)

#     def __getitem__(self, i):
#         if self.shuffle and i == 0:
#             self.indexes = self.np_rng.permutation(self.indexes)
#         start = i * self.batch_size
#         end = (i + 1) * self.batch_size
#         data_batch = np.empty((self.batch_size, *self.data_shape))

#         for j in range(start, end):
#             data_ = pv.read(self.filenames[self.indexes[j]])
#             data_ = data_.get_array("-velocity_magnitude")
#             data_ = data_.reshape(self.data_shape)
#             data_ = self.normalize(data_)
#             data_batch[j - start] = data_

#         return (data_batch, data_batch)

#     def normalize(self, data):
#         return data / 6.8

#     def denormalize(self, data):
#         return data * 6.8

#     def __len__(self):
#         return len(self.indexes) // self.batch_size


# class FlameGenerator(tf.keras.utils.PyDataset):
#     def __init__(
#         self,
#         data_dir,
#         batch_size,
#         data_shape,
#         shuffle=False,
#         workers=4,
#         use_multiprocessing=True,
#         max_queue_size=10,
#         eval_mode=False,
#         seed=0,
#         **kwargs,
#     ):
#         super().__init__(
#             workers=workers,
#             use_multiprocessing=use_multiprocessing,
#             max_queue_size=max_queue_size,
#             **kwargs,
#         )
#         self.data_dir = data_dir
#         self.batch_size = batch_size
#         self.data_shape = data_shape
#         # self.shuffle = shuffle
#         # self.np_rng = np.random.default_rng(seed)

#         self.filenames = glob.glob(data_dir + "/*.vtk", recursive=True)
#         self.filenames.sort(
#             key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1])
#         )
#         # self.filenames = self.filenames[:88]
#         self.num_files = len(self.filenames)

#         if not eval_mode:
#             self.num_files = int(self.num_files * 0.9)
#             self.filenames = self.filenames[: self.num_files]
#         else:
#             self.num_files = self.num_files - int(self.num_files * 0.9)
#             self.filenames = self.filenames[self.num_files :]

#     def __getitem__(self, i):
#         data_ = pv.read(self.filenames[i])
#         data_ = data_.get_array("-velocity_magnitude")
#         data_ = data_.reshape(self.data_shape)
#         data_ = self.normalize(data_)
#         return (data_, data_)

#         # if self.shuffle and i == 0:
#         #     self.indexes = self.np_rng.permutation(self.indexes)
#         # start = i * self.batch_size
#         # end = (i + 1) * self.batch_size
#         # data_batch = np.empty((self.batch_size, *self.data_shape))

#         # for j in range(start, end):
#         #     data_ = pv.read(self.filenames[self.indexes[j]])
#         #     data_ = data_.get_array("-velocity_magnitude")
#         #     data_ = data_.reshape(self.data_shape)
#         #     data_ = self.normalize(data_)
#         #     data_batch[j - start] = data_

#         # return (data_batch, data_batch)

#     def normalize(self, data):
#         return data / 6.8

#     def denormalize(self, data):
#         return data * 6.8

#     def __len__(self):
#         return len(self.indexes) // self.batch_size


######################################################################
# Pytorch DataLoader
######################################################################


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def create_data_loaders(
    *datasets: Sequence[torch.utils.data.Dataset],
    train: Union[bool, Sequence[bool]] = True,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 42,
):
    """
    Creates data loaders used in JAX for a set of datasets.

    Args:
      datasets: Datasets for which data loaders are created.
      train: Sequence indicating which datasets are used for
        training and which not. If single bool, the same value
        is used for all datasets.
      batch_size: Batch size to use in the data loaders.
      num_workers: Number of workers for each dataset.
      seed: Seed to initialize the workers and shuffling with.
    """
    loaders = []
    if not isinstance(train, (list, tuple)):
        train = [train for _ in datasets]
    for dataset, is_train in zip(datasets, train):
        if not is_train:
            num_workers_ = (num_workers // 4) or 1
        else:
            num_workers_ = num_workers
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            drop_last=is_train,
            collate_fn=numpy_collate,
            num_workers=num_workers_,
            # persistent_workers=is_train,
            generator=torch.Generator().manual_seed(seed),
        )
        loaders.append(loader)
    return loaders


class FlameGenerator(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        data_shape: Sequence[int],
        is_train: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data_shape = data_shape

        self.filenames = glob.glob(data_dir + "/*.vtk", recursive=True)
        self.filenames.sort(
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1])
        )
        # self.filenames = self.filenames[:88]
        self.num_files = len(self.filenames)

        if is_train:
            self.num_files = int(self.num_files * 0.9)
            self.filenames = self.filenames[: self.num_files]
        else:
            self.num_files = int(self.num_files * 0.1)
            self.filenames = self.filenames[self.num_files :]

    def __getitem__(self, i):
        data_ = pv.read(self.filenames[i])
        data_ = data_.get_array("-velocity_magnitude")
        data_ = np.array(data_)
        data_ = data_.reshape(self.data_shape)
        data_ = self.normalize(data_)
        return (data_, data_)

    def normalize(self, data):
        return data / 6.8

    def denormalize(self, data):
        return data * 6.8

    def __len__(self):
        return self.num_files


class SequenceGenerator(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        data_shape: Sequence[int],
        is_train: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data_shape = data_shape

        self.filenames = glob.glob(data_dir + "/*.vtk", recursive=True)
        self.filenames.sort(
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1])
        )
        # self.filenames = self.filenames[:88]
        self.num_files = len(self.filenames)

        if is_train:
            self.num_files = int(self.num_files * 0.9)
            self.filenames = self.filenames[: self.num_files]
        else:
            self.num_files = int(self.num_files * 0.1)
            self.filenames = self.filenames[self.num_files :]

    def read_vtk(self, filename):
        data_ = pv.read(filename)
        data_ = data_.get_array("-velocity_magnitude")
        data_ = np.array(data_)
        data_ = data_.reshape(self.data_shape)
        data_ = self.normalize(data_)
        return data_

    def __getitem__(self, i):
        input_ = self.read_vtk(self.filenames[i])
        output_ = self.read_vtk(self.filenames[i + 1])
        return (input_, output_)

    def normalize(self, data):
        return data / 6.8

    def denormalize(self, data):
        return data * 6.8

    def __len__(self):
        return self.num_files - 1
