"""Module with augmentations retrieving"""

from albumentations import (
    Compose,
    GaussNoise,
    HorizontalFlip,
    Normalize,
    ShiftScaleRotate,
    ToGray,
)
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transforms() -> Compose:

    transforms = [
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30),
        HorizontalFlip(p=0.5),
        GaussNoise(p=0.3),
        Normalize(mean=0, std=1, always_apply=True),
        ToTensorV2(always_apply=True),
    ]
    transforms = Compose(transforms)

    return transforms


def get_val_transforms() -> Compose:
    transforms = [
        ToGray(),
        Normalize(mean=0, std=1, always_apply=True),
        ToTensorV2(always_apply=True),
    ]
    transforms = Compose(transforms)

    return transforms
