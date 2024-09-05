from typing import Sequence, Union
import torch
import torch.utils.data
import numpy as np
import glob
import os

# import transform
from torchvision.transforms import v2

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
            # collate_fn=numpy_collate,
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

        self.filenames = glob.glob(data_dir + "/*.npy", recursive=True)
        self.num_files = len(self.filenames)
        self.np_data = np.load(self.filenames[0])
        mean = np.mean(self.np_data)
        std = np.std(self.np_data)
        print(f"mean: {np.mean(self.np_data)}, std: {np.std(self.np_data)}")
        print(f"min: {np.min(self.np_data)}, max: {np.max(self.np_data)}")

        if is_train:
            self.np_data = self.np_data[:, : int(self.np_data.shape[1] * 0.9)]
        else:
            self.np_data = self.np_data[:, int(self.np_data.shape[1] * 0.9) :]

        self.transforms = v2.Compose(
            [
                v2.Normalize(mean=[mean], std=[std]),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def __getitem__(self, i):
        data_ = self.np_data[:, i : i + 1]
        data_ = np.swapaxes(data_, 0, 1)
        data_ = self.transforms(data_)
        return (data_, data_)

    def __len__(self):
        return self.np_data.shape[1]


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

        self.filenames = glob.glob(data_dir + "/*.npy", recursive=True)
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


class FlameGeneratorVTK(torch.utils.data.Dataset):
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


class SequenceGeneratorVTK(torch.utils.data.Dataset):
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
