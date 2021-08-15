"""Module with utils for data"""

import glob
import os
import random
from typing import List, Tuple

from torch.utils.data import DataLoader, Dataset

from blink_classification.modules.data.augmentations import (
    get_train_transforms,
    get_val_transforms,
)
from blink_classification.modules.data.dataset import BlinkDataset


def get_train_val_datasets(
    data_folder: str,
    image_size: Tuple[int, int] = (128, 128),
    valid_percent: float = 0.3,
) -> Tuple[Dataset, Dataset]:
    train_list_of_paths, val_list_of_paths = split_list_of_paths(
        data_folder=data_folder, valid_percent=valid_percent
    )

    train_dataset = BlinkDataset(
        list_of_paths=train_list_of_paths,
        image_size=image_size,
        transforms=get_train_transforms(),
    )
    val_dataset = BlinkDataset(
        list_of_paths=val_list_of_paths,
        image_size=image_size,
        transforms=get_val_transforms(),
    )

    return train_dataset, val_dataset


def create_data_loader(
    dataset: Dataset, batch_size: int = 1, shuffle: bool = True, num_workers: int = 2
) -> DataLoader:
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return data_loader


def split_list_of_paths(
    data_folder: str, valid_percent: float = 0.3
) -> Tuple[List[str], List[str]]:

    list_of_paths = get_list_of_paths(data_folder=data_folder)
    random.shuffle(list_of_paths)

    edge_value = int(len(list_of_paths) * valid_percent)

    train_list_of_paths = list_of_paths[:-edge_value]
    val_list_of_paths = list_of_paths[-edge_value:]

    return train_list_of_paths, val_list_of_paths


def get_list_of_paths(
    data_folder: str, image_file_template: str = '**/*.png'
) -> List[str]:
    image_path_template = os.path.join(data_folder, image_file_template)

    list_of_paths = glob.glob(pathname=image_path_template, recursive=True)

    return list_of_paths
