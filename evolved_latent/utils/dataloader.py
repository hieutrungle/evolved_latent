import tensorflow as tf
import pyvista as pv
import numpy as np
import glob
import os


class FlameGenerator(tf.keras.utils.PyDataset):
    def __init__(
        self,
        data_dir,
        batch_size,
        data_shape,
        shuffle=True,
        workers=4,
        use_multiprocessing=True,
        max_queue_size=10,
        **kwargs
    ):
        super().__init__(
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            max_queue_size=max_queue_size,
            **kwargs
        )
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.shuffle = shuffle

        self.filenames = glob.glob(data_dir + "/*.vtk", recursive=True)
        self.filenames.sort(
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1])
        )
        self.num_files = len(self.filenames)
        self.indexes = np.arange(self.num_files)

    def __getitem__(self, i):
        start = i * self.batch_size
        end = (i + 1) * self.batch_size
        data_batch = np.empty((self.batch_size, *self.data_shape))

        for j in range(start, end):
            data_ = pv.read(self.filenames[j])
            data_ = data_.get_array("-velocity_magnitude")
            data_ = data_.reshape(self.data_shape)
            data_batch[j - start] = data_

        return (data_batch, data_batch)

    def __len__(self):
        return len(self.indexes) // self.batch_size
