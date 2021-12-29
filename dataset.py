import numpy as np

from torchvision.datasets import STL10, CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader


class MyValSet(Dataset):
    def __init__(self, TRAIN_DATASET, TEST_DATASET):
        self.train_size = len(TRAIN_DATASET)
        self.TRAIN_DATASET = TRAIN_DATASET
        self.TEST_DATASET = TEST_DATASET

    def __len__(self):
        return len(self.TRAIN_DATASET) + len(self.TEST_DATASET)

    def __getitem__(self, idx):
        if idx < self.train_size:
            image = self.TRAIN_DATASET.data[idx]
            label = self.TRAIN_DATASET.labels[idx]
            from_train = True  # fine-tuning train data
        else:
            image = self.TEST_DATASET.data[idx]
            label = self.TEST_DATASET.labels[idx]
            from_train = False  # fine-tuning val dat
        return image.astype(np.float32), label.astype(np.float32), from_train


def adjust_dataset_size(ds, ratio):
    size = int(len(ds) * ratio)
    ds.data = ds.data[:size]
    ds.labels = ds.labels[:size]
    return ds


def prep_data(args):
    """
    stl_10 dataset
        Train:5k,
        Unlabel:100k,
        Test:8k
    """
    batch_size = args.bz
    TRAIN_DATASET = STL10(root="data", split="train",
                          download=True, transform=ToTensor())
    TEST_DATASET = STL10(root="data", split="test",
                         download=True, transform=ToTensor())

    BYOL_TRAIN_DATASET = STL10(
        root="data", split="train+unlabeled", download=True, transform=ToTensor())
    # Previously, we actually split test data in half, each for training and the other for val
    # This byol_val dataset enables us to use the same train & val dataset as supervise-learning.
    BYOL_VAL_DATASET = MyValSet(TRAIN_DATASET, TEST_DATASET)

    FINETUNE_TRAIN_DATASET = STL10(
        root="data", split="train", download=True, transform=ToTensor())

    # adjst dataset size
    BYOL_TRAIN_DATASET = adjust_dataset_size(
        BYOL_TRAIN_DATASET, ratio=args.byol_ratio)
    FINETUNE_TRAIN_DATASET = adjust_dataset_size(
        FINETUNE_TRAIN_DATASET, ratio=args.ft_ratio)

    train_loader = DataLoader(
        TRAIN_DATASET,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        TEST_DATASET,
        batch_size=batch_size,
        num_workers=4
    )
    byol_train_loader = DataLoader(
        BYOL_TRAIN_DATASET,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )
    byol_val_loader = DataLoader(
        BYOL_VAL_DATASET,
        batch_size=batch_size,
        num_workers=4
    )
    finetune_train_loader = DataLoader(
        FINETUNE_TRAIN_DATASET,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    return train_loader, val_loader, byol_train_loader, byol_val_loader, finetune_train_loader


def prep_data_cifar(args):  # using STL10 to pretrain for CIFAR
    batch_size = args.bz
    TRAIN_DATASET = CIFAR10(root="data", split="train",
                            download=True, transform=ToTensor())
    TEST_DATASET = CIFAR10(root="data", split="test",
                           download=True, transform=ToTensor())

    BYOL_TRAIN_DATASET = STL10(
        root="data", split="train+unlabeled", download=True, transform=ToTensor())
    # Previously, we actually split test data in half, each for training and the other for val
    # This byol_val dataset enables us to use the same train & val dataset as supervise-learning.
    BYOL_VAL_DATASET = MyValSet(TRAIN_DATASET, TEST_DATASET)

    train_loader = DataLoader(
        TRAIN_DATASET,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        TEST_DATASET,
        batch_size=batch_size,
        num_workers=4
    )
    byol_train_loader = DataLoader(
        BYOL_TRAIN_DATASET,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )
    byol_val_loader = DataLoader(
        BYOL_VAL_DATASET,
        batch_size=batch_size,
        num_workers=4
    )

    return train_loader, val_loader, byol_train_loader, byol_val_loader
