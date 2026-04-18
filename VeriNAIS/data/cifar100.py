"""
data/cifar100.py
----------------
CIFAR-100 dataset loader for STL-NAS experiments.

CIFAR-100 is used in the temporal property ablation study (Section VII-F /
Table 5) and for the Phase 2E extended temporal property experiments.
It has the same 32×32 image format as CIFAR-10 but with 100 classes grouped
into 20 superclasses.

Dataset statistics
------------------
  Training set:   50 000 images (32×32 RGB), 100 balanced classes
  Test set:       10 000 images
  Mean:           (0.5071, 0.4867, 0.4408)
  Std:            (0.2675, 0.2565, 0.2761)
  Top-1 accuracy (NAS-Bench-201 best arch):  ~73.5 %

Download and load CIFAR-100
-----------------------------
CIFAR-100 is available via torchvision.datasets.CIFAR100.

  from torchvision import datasets, transforms

  transform_train = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.5071, 0.4867, 0.4408),
                           (0.2675, 0.2565, 0.2761)),
  ])

  transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5071, 0.4867, 0.4408),
                           (0.2675, 0.2565, 0.2761)),
  ])

  train_dataset = datasets.CIFAR100(
      root='data/cifar100', train=True,
      download=True, transform=transform_train,
  )
  test_dataset = datasets.CIFAR100(
      root='data/cifar100', train=False,
      download=True, transform=transform_test,
  )
"""

from __future__ import annotations

from typing import Tuple

# Install torchvision: pip install torchvision
# import torch
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms


# Dataset statistics
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)
NUM_CLASSES   = 100
IMAGE_SIZE    = (3, 32, 32)


def get_cifar100_loaders(
    data_root: str = "data/cifar100",
    batch_size_train: int = 128,
    batch_size_test: int = 256,
    num_workers: int = 4,
    augment: bool = True,
    download: bool = True,
) -> Tuple["DataLoader", "DataLoader"]:
    """Return (train_loader, test_loader) for CIFAR-100.

    Parameters
    ----------
    data_root :
        Directory where CIFAR-100 will be stored / is already stored.
        Downloaded automatically if download=True.
    batch_size_train :
        Batch size for the training loader.
    batch_size_test :
        Batch size for the test loader.
    num_workers :
        Number of worker processes for data loading.
    augment :
        If True, apply RandomCrop + RandomHorizontalFlip to training data.
    download :
        If True, download CIFAR-100 if not already present.

    Returns
    -------
    (train_loader, test_loader)
    """
    # ── Download and load CIFAR-100 ────────────────────────────────────────
    # from torchvision import datasets, transforms
    # from torch.utils.data import DataLoader
    #
    # norm = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    #
    # train_transforms = [transforms.ToTensor(), norm]
    # if augment:
    #     train_transforms = [
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #     ] + train_transforms
    #
    # train_dataset = datasets.CIFAR100(
    #     root=data_root, train=True,
    #     download=download,
    #     transform=transforms.Compose(train_transforms),
    # )
    # test_dataset = datasets.CIFAR100(
    #     root=data_root, train=False,
    #     download=download,
    #     transform=transforms.Compose([transforms.ToTensor(), norm]),
    # )
    #
    # train_loader = DataLoader(
    #     train_dataset, batch_size=batch_size_train,
    #     shuffle=True, num_workers=num_workers, pin_memory=True,
    # )
    # test_loader = DataLoader(
    #     test_dataset, batch_size=batch_size_test,
    #     shuffle=False, num_workers=num_workers, pin_memory=True,
    # )
    # return train_loader, test_loader
    raise NotImplementedError(
        "Install torchvision (pip install torchvision) and uncomment the "
        "implementation above to load CIFAR-100."
    )
