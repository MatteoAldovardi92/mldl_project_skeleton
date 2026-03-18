import torch
from torch.utils.data import DataLoader
# Import the datasets defined in your utils/data.py
from utils.data import tiny_imagenet_dataset_train, tiny_imagenet_dataset_val

def get_dataloaders(batch_size=64, num_workers=2):
    """
    Creates DataLoader instances for the Tiny ImageNet training and validation sets.
    """
    train_loader = DataLoader(
        tiny_imagenet_dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        tiny_imagenet_dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
