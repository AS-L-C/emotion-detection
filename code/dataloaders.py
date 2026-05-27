import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


class EEGDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_balanced_split_indices(labels, seed, f_train):
    labels_np = np.asarray(labels)
    rng = np.random.default_rng(seed)

    train_idx = []
    valid_idx = []

    for cl in np.unique(labels_np):
        cls_idx = np.where(labels_np == cl)[0]
        rng.shuffle(cls_idx)

        n_cls = len(cls_idx)
        n_train_cls = round(f_train * n_cls)

        train_idx.extend(cls_idx[:n_train_cls].tolist())
        valid_idx.extend(cls_idx[n_train_cls:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(valid_idx)

    return np.array(train_idx), np.array(valid_idx)


def _make_random_split_indices(n, seed, f_train):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_train = int(round(f_train * n))

    train_idx = indices[:n_train]
    valid_idx = indices[n_train:]

    return train_idx, valid_idx


def make_dataloaders(
    data,
    labels,
    batch_size=64,
    seed=0,
    f_train=0.8,
    f_valid=0.2,
    shuffle=True,
    balanced=True,
):
    # Assertions
    assert "train" in data, "data must contain a 'train' split"
    assert "test" in data, "data must contain a 'test' split"
    assert abs((f_train + f_valid) - 1.0) < 1e-8, "f_train + f_valid must equal 1"

    # Initializations
    splits = ["train", "valid", "test"]
    datasets = {s: None for s in splits}
    dataloaders = {s: None for s in splits}

    # Conversions to tensors
    x_train_all = torch.as_tensor(data["train"], dtype=torch.float32)
    x_test = torch.as_tensor(data["test"], dtype=torch.float32)
    y_train_all = torch.as_tensor(labels, dtype=torch.long)
    n_train_all = len(x_train_all)

    # Build split indices
    if balanced:
        train_idx, valid_idx = _make_balanced_split_indices(
            y_train_all.numpy(),
            seed=seed,
            f_train=f_train,
        )
    else:
        train_idx, valid_idx = _make_random_split_indices(
            n=n_train_all,
            seed=seed,
            f_train=f_train,
        )

    # Build training and validation datasets
    train_dataset_full = EEGDataset(x_train_all, y_train_all)
    datasets["train"] = Subset(train_dataset_full, train_idx.tolist())
    datasets["valid"] = Subset(train_dataset_full, valid_idx.tolist())

    # Build test dataset
    # Use dummy label -1 for test dataset
    y_test_dummy = torch.full((len(x_test),), -1, dtype=torch.long)
    datasets["test"] = EEGDataset(x_test, y_test_dummy)

    # Build dataloaders
    for isplit, split in enumerate(splits):
        dataloaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=shuffle if split == "train" else False,
            generator=torch.Generator().manual_seed(seed + isplit),
            worker_init_fn=_seed_worker,
            drop_last=False,
        )
    return dataloaders
