"""Module with augmentations retrieving"""

from typing import Tuple

from albumentations import Compose, GaussNoise, HorizontalFlip, Normalize, Resize
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transforms(image_size: Tuple[int, int]) -> Compose:

    transforms = [
        Resize(height=image_size[0], width=image_size[1], always_apply=True),
        HorizontalFlip(p=0.5),
        GaussNoise(p=0.3),
        Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True
        ),
        ToTensorV2(always_apply=True),
    ]
    transforms = Compose(transforms)

    return transforms


def get_val_transforms(image_size: Tuple[int, int]) -> Compose:
    transforms = [
        Resize(height=image_size[0], width=image_size[1], always_apply=True),
        Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True
        ),
        ToTensorV2(always_apply=True),
    ]
    transforms = Compose(transforms)

    return transforms
