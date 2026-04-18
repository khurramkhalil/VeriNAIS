"""
data/cifar10.py
---------------
CIFAR-10 dataset loader for STL-NAS experiments.

In the NAS-Bench-201 experimental setting (Sections VII-B through VII-F),
accuracy values are retrieved directly from the pre-computed benchmark table
(200-epoch training) rather than by training models from scratch.  This
module is therefore used only for:

  (a) Constructing the proxy model calibration images fed to TensorRT
      during Jetson Thor latency profiling (Phase 2B).
  (b) Validation-set evaluation when running NAS paradigms outside the
      NAS-Bench-201 controlled setting (e.g. full training ablations).

Dataset statistics
------------------
  Training set:   50 000 images (32×32 RGB), 10 balanced classes
  Test set:       10 000 images
  Mean:           (0.4914, 0.4822, 0.4465)
  Std:            (0.2023, 0.1994, 0.2010)
  Classes:        airplane, automobile, bird, cat, deer,
                  dog, frog, horse, ship, truck

Download and load CIFAR-10
---------------------------
CIFAR-10 is available via torchvision.datasets.CIFAR10.

  from torchvision import datasets, transforms

  transform_train = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010)),
  ])

  transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010)),
  ])

  train_dataset = datasets.CIFAR10(
      root='data/cifar10', train=True,
      download=True, transform=transform_train,
  )
  test_dataset = datasets.CIFAR10(
      root='data/cifar10', train=False,
      download=True, transform=transform_test,
  )

  train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                            num_workers=4, pin_memory=True)
  test_loader  = DataLoader(test_dataset,  batch_size=256, shuffle=False,
                            num_workers=4, pin_memory=True)
"""

from __future__ import annotations

from typing import Optional, Tuple

# torchvision is required for dataset download and augmentation.
# Install with: pip install torchvision
# import torch
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms


# Dataset statistics (used for normalisation)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
NUM_CLASSES  = 10
IMAGE_SIZE   = (3, 32, 32)   # C × H × W


def get_cifar10_loaders(
    data_root: str = "data/cifar10",
    batch_size_train: int = 128,
    batch_size_test: int = 256,
    num_workers: int = 4,
    augment: bool = True,
    download: bool = True,
) -> Tuple["DataLoader", "DataLoader"]:
    """Return (train_loader, test_loader) for CIFAR-10.

    Parameters
    ----------
    data_root :
        Directory where CIFAR-10 will be stored / is already stored.
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
        If True, download CIFAR-10 if not already present.

    Returns
    -------
    (train_loader, test_loader)
    """
    # ── Download and load CIFAR-10 ─────────────────────────────────────────
    # from torchvision import datasets, transforms
    # from torch.utils.data import DataLoader
    #
    # norm = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    #
    # train_transforms = [transforms.ToTensor(), norm]
    # if augment:
    #     train_transforms = [
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #     ] + train_transforms
    #
    # train_dataset = datasets.CIFAR10(
    #     root=data_root, train=True,
    #     download=download,
    #     transform=transforms.Compose(train_transforms),
    # )
    # test_dataset = datasets.CIFAR10(
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
        "implementation above to load CIFAR-10."
    )


def get_single_batch(
    data_root: str = "data/cifar10",
    batch_size: int = 1,
) -> "torch.Tensor":
    """Return a single input batch for TensorRT profiling on Jetson Thor.

    Used by the measure_harness to obtain a representative inference input
    for latency and power measurement.

    Parameters
    ----------
    data_root :
        Path to CIFAR-10 data.
    batch_size :
        Number of images (1 for single-image latency measurement).

    Returns
    -------
    torch.Tensor
        Normalised input tensor of shape (batch_size, 3, 32, 32) on CPU.
    """
    # ── Load a single normalised CIFAR-10 batch ────────────────────────────
    # from torchvision import datasets, transforms
    # from torch.utils.data import DataLoader
    #
    # norm = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    # dataset = datasets.CIFAR10(
    #     root=data_root, train=False,
    #     download=True,
    #     transform=transforms.Compose([transforms.ToTensor(), norm]),
    # )
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # images, _ = next(iter(loader))
    # return images
    raise NotImplementedError(
        "Install torchvision and uncomment the implementation above."
    )
