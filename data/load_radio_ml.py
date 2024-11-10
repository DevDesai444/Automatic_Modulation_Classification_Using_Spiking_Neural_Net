import os
import h5py
import torch
import numpy as np
from torch.utils import data
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader


class RadioMLDataset(data.Dataset):
    """RadioML data.
    Available here: https://www.deepsig.io/datasets.
    """

    def __init__(self, data_dir, train,
                 normalize=False, min_snr=6, max_snr=30, per_h5_frac=0.5, train_frac=0.9):

        self.train = train

        if not os.path.exists(os.path.join(data_dir, 'class23_snr30.hdf5')):
            # Split huge HDF5 file into per-SNR, per-class HDF5 files

            data_path = os.path.join(data_dir, 'GOLD_XYZ_OSC.0001_1024.hdf5')
            full_h5f = h5py.File(data_path, 'r')
            Y = np.argmax(full_h5f['Y'], axis=1)

            for class_idx in range(24):
                class_X = full_h5f['X'][Y == class_idx, :, :]
                class_Z = full_h5f['Z'][Y == class_idx, 0]
                for snr in range(-26, 32, 2):
                    class_snr_name = 'class%d_snr%d.hdf5' % (class_idx, snr)
                    h5f_path = os.path.join(data_dir, class_snr_name)
                    h5f = h5py.File(h5f_path, 'w')
                    h5f.create_dataset('X', data=class_X[class_Z == snr, :, :])
                    h5f.close()
                    print('Wrote (SNR {z}, class {cl}) data to `{path}`.'.format(
                        z=snr, cl=class_idx, path=h5f_path))
                class_X = None
                class_Z = None
            Y = None
            full_h5f.close()
        X_minval = float('inf')
        X_maxval = float('-inf')
        per_h5_size = int(per_h5_frac * 4096)
        snr_count = (max_snr - min_snr) // 2 + 1
        train_split_size = int(train_frac * per_h5_size)
        if train:
            split_size = train_split_size
        else:
            split_size = per_h5_size - train_split_size
        total_size = 2 * snr_count * split_size

        self.X = np.zeros((total_size, 1024, 2), dtype=np.float32)
        self.Y = np.zeros(total_size, dtype=np.int64)
        for class_idx in range(2):
            for snr_idx, snr in enumerate(range(min_snr, max_snr + 2, 2)):
                class_snr_name = 'class%d_snr%d.hdf5' % (class_idx, snr)
                h5f_path = os.path.join(data_dir, class_snr_name)
                h5f = h5py.File(h5f_path, 'r')
                X = h5f['X'][:]
                X_minval = min(X_minval, X.min())
                X_maxval = max(X_maxval, X.max())
                if train:
                    X_split = X[:train_split_size]
                else:
                    X_split = X[train_split_size:per_h5_size]
                # Interleave
                start_idx = (class_idx * snr_count) + snr_idx
                self.X[start_idx::2*snr_count] = X_split
                self.Y[start_idx::2*snr_count] = class_idx
                h5f.close()
                X = None
                X_split = None
        print(f'----------> Shape before transpose: {self.X.shape}')
        self.X = self.X.transpose(0, 2, 1)[:, :, np.newaxis, :]
        print(f'----------> Shape after transpose: {self.X.shape}')
        if normalize:
            self.X = (self.X - X_minval) / (X_maxval - X_minval)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def get_radio_ml_loader(batch_size, train, **kwargs):
    data_dir = kwargs['data_dir']
    min_snr = kwargs.get('min_snr', 6)
    max_snr = kwargs.get('max_snr', 30)
    per_h5_frac = kwargs.get('per_h5_frac', 0.5)
    train_frac = kwargs.get('train_frac', 0.9)
    dataset = RadioMLDataset(data_dir, train,
                             normalize=False,
                             min_snr=min_snr,
                             max_snr=max_snr,
                             per_h5_frac=per_h5_frac,
                             train_frac=train_frac)

    identifier = 'train' if train else 'test'
    print('[%s] dataset size: %d -> each file has shape: %s , %s' % (identifier, len(dataset), dataset[0][0].shape , dataset[13][1]))

    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=train)
    loader.name = 'RadioML_{}'.format(identifier)

    return loader
