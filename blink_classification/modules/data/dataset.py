"""Module with dataset code"""


import glob
import os
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from cv2 import cv2
from torch.utils.data import Dataset


class BlinkDataset(Dataset):
    EYE_STATE_IDX = 4

    def __init__(
        self,
        data_folder: str,
        image_size: Tuple[int, int] = (128, 128),
        transforms: Compose = None,
    ) -> None:

        self.list_of_paths = self.__get_list_of_paths(data_folder=data_folder)

        self.image_size = image_size
        self.transforms = transforms

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, int]]:
        curr_image_path = self.list_of_paths[idx]

        image: np.ndarray = cv2.imread(filename=curr_image_path)
        eye_state: int = self.__get_eye_state(image_path=curr_image_path)

        if self.transforms:
            image_tensor: torch.Tensor = self.transforms(image=image)['image']
        else:
            image = self.__resize_image(image=image, image_size=self.image_size)
            image_tensor = self.__to_tensor(image=image)

        dataset_item = {'image': image_tensor, 'label': eye_state}

        return dataset_item

    def __len__(self) -> int:
        length = len(self.list_of_paths)

        return length

    def __get_eye_state(self, image_path: str) -> int:
        image_filename = image_path.split(os.sep)[-1]
        eye_state: str = image_filename.split("_")[self.EYE_STATE_IDX]

        try:
            eye_state_number: int = int(eye_state)
        except ValueError:
            raise ValueError(
                f'Bad file was passed ({image_filename}).'
                f' Eye state parsed: {eye_state}'
            )

        return eye_state_number

    @staticmethod
    def __to_tensor(image: np.ndarray) -> torch.Tensor:
        transform_to_tensor = ToTensorV2(always_apply=True)
        image = transform_to_tensor(image=image)['image']

        return image

    @staticmethod
    def __resize_image(image: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
        image = cv2.resize(src=image, dsize=image_size)

        return image

    @staticmethod
    def __get_list_of_paths(data_folder: str) -> List[str]:
        image_file_template: str = '**/*.png'
        image_path_template = os.path.join(data_folder, image_file_template)

        list_of_paths = glob.glob(pathname=image_path_template, recursive=True)

        return list_of_paths
