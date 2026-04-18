"""
data/imagenet16.py
------------------
ImageNet-16-120 dataset loader for STL-NAS experiments.

ImageNet-16-120 is a downsampled subset of ImageNet (16×16 pixels, 120
classes) used in NAS-Bench-201 as a harder benchmark than CIFAR-10/100.
It is used in Table 5 of the paper for the cross-dataset generalisation
experiments.

Dataset statistics
------------------
  Training set:   151 700 images (16×16 RGB), 120 classes
  Validation set: 3 000 images
  Test set:       6 000 images
  Mean:           (0.4811, 0.4575, 0.4078)
  Std:            (0.2604, 0.2532, 0.2686)
  Top-1 accuracy (NAS-Bench-201 best arch):  ~47.3 %

Download instructions
---------------------
ImageNet-16-120 is distributed with the NAS-Bench-201 repository.

  1. Download ImageNet16.tar from:
       https://drive.google.com/drive/folders/1T3UIyZXUhMmIuJLOBMIYKAsJknAtrrO2
     or via the NATS-Bench data utilities.

  2. Extract to data/imagenet16/:
       tar -xf ImageNet16.tar -C data/imagenet16/

  3. The loader reads the per-class .bin files using the ImageNet16 class
     from the NAS-Bench-201 / NATS-Bench data utilities.

Loading CIFAR-style with the NAS-Bench-201 ImageNet16 helper
-------------------------------------------------------------
  from nats_bench.data_utils.imagenet16 import ImageNet16

  train_dataset = ImageNet16(
      root='data/imagenet16',
      train=True,
      transform=train_transform,
      use_num_of_class_only=120,
  )
  test_dataset = ImageNet16(
      root='data/imagenet16',
      train=False,
      transform=test_transform,
      use_num_of_class_only=120,
  )
"""

from __future__ import annotations

from typing import Tuple

# Install torchvision and the NATS-Bench data utilities:
#   pip install torchvision
#   git clone https://github.com/D-X-Y/NATS-Bench third_party/nats_bench
# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from third_party.nats_bench.data_utils.imagenet16 import ImageNet16


# Dataset statistics
IMAGENET16_MEAN = (0.4811, 0.4575, 0.4078)
IMAGENET16_STD  = (0.2604, 0.2532, 0.2686)
NUM_CLASSES     = 120
IMAGE_SIZE      = (3, 16, 16)


def get_imagenet16_loaders(
    data_root: str = "data/imagenet16",
    batch_size_train: int = 256,
    batch_size_test: int = 256,
    num_workers: int = 4,
    augment: bool = True,
    n_classes: int = 120,
) -> Tuple["DataLoader", "DataLoader"]:
    """Return (train_loader, test_loader) for ImageNet-16-120.

    Parameters
    ----------
    data_root :
        Directory containing the ImageNet-16 .bin files.
        Extract ImageNet16.tar to this location.
    batch_size_train :
        Batch size for the training loader.
    batch_size_test :
        Batch size for the test/validation loader.
    num_workers :
        Number of data-loading workers.
    augment :
        Apply RandomCrop (16, padding=2) + RandomHorizontalFlip during training.
    n_classes :
        Number of classes to use (120 for ImageNet-16-120).

    Returns
    -------
    (train_loader, test_loader)
    """
    # ── Download and load ImageNet-16-120 ──────────────────────────────────
    # from torchvision import transforms
    # from torch.utils.data import DataLoader
    # from third_party.nats_bench.data_utils.imagenet16 import ImageNet16
    #
    # norm = transforms.Normalize(IMAGENET16_MEAN, IMAGENET16_STD)
    #
    # train_transforms = [transforms.ToTensor(), norm]
    # if augment:
    #     train_transforms = [
    #         transforms.RandomCrop(16, padding=2),
    #         transforms.RandomHorizontalFlip(),
    #     ] + train_transforms
    #
    # train_dataset = ImageNet16(
    #     root=data_root, train=True,
    #     transform=transforms.Compose(train_transforms),
    #     use_num_of_class_only=n_classes,
    # )
    # test_dataset = ImageNet16(
    #     root=data_root, train=False,
    #     transform=transforms.Compose([transforms.ToTensor(), norm]),
    #     use_num_of_class_only=n_classes,
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
        "Download ImageNet16.tar from the NAS-Bench-201 data utilities, "
        "extract to data/imagenet16/, and uncomment the implementation above."
    )
