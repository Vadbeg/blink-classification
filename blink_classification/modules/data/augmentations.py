"""Module with augmentations retrieving"""

from albumentations import Compose, GaussNoise, HorizontalFlip, Normalize
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transforms() -> Compose:

    transforms = [
        HorizontalFlip(p=0.5),
        GaussNoise(p=0.3),
        Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True
        ),
        ToTensorV2(always_apply=True),
    ]
    transforms = Compose(transforms)

    return transforms


def get_val_transforms() -> Compose:
    transforms = [
        Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True
        ),
        ToTensorV2(always_apply=True),
    ]
    transforms = Compose(transforms)

    return transforms
