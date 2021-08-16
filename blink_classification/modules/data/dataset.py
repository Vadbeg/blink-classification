"""Module with dataset code"""


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
        list_of_paths: List[str],
        image_size: Tuple[int, int] = (128, 128),
        transforms: Compose = None,
        is_eval: bool = False,
    ) -> None:

        self.list_of_paths = list_of_paths

        self.image_size = image_size
        self.transforms = transforms

        self.is_eval = is_eval

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int]]:
        curr_image_path = self.list_of_paths[idx]

        image: np.ndarray = self.__load_and_preprocess_image(image_path=curr_image_path)

        eye_state: int = -1
        if not self.is_eval:
            eye_state = self.__get_eye_state(image_path=curr_image_path)

        image = self.__resize_image(image=image, image_size=self.image_size)

        if self.transforms:
            image_tensor: torch.Tensor = self.transforms(image=image)['image']
        else:
            image_tensor = self.__to_tensor(image=image)

        dataset_item = {'image': image_tensor, 'label': eye_state}

        return dataset_item

    def __len__(self) -> int:
        length = len(self.list_of_paths)

        return length

    def __get_eye_state(self, image_path: str) -> int:
        image_filename = image_path.split(os.sep)[-1]
        image_params: List[str] = image_filename.split("_")

        if len(image_params) <= self.EYE_STATE_IDX:
            raise IndexError(f'No params in filename: {image_filename}')

        eye_state: str = image_params[self.EYE_STATE_IDX]

        try:
            eye_state_number: int = int(eye_state)
        except ValueError:
            raise ValueError(
                f'Bad file was passed ({image_filename}).'
                f' Eye state parsed: {eye_state}'
            )

        return eye_state_number

    def __load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        image: np.ndarray = cv2.imread(filename=image_path)
        image = self.__resize_image(image=image, image_size=self.image_size)

        if not self.__check_if_already_gray(image=image):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return image

    @staticmethod
    def __check_if_already_gray(image: np.ndarray) -> bool:
        is_gray = False

        if len(image.shape) < 3:
            is_gray = True
        elif len(image.shape) == 3 and image.shape[-1] == 1:
            is_gray = True

        return is_gray

    @staticmethod
    def __to_tensor(image: np.ndarray) -> torch.Tensor:
        transform_to_tensor = ToTensorV2(always_apply=True)
        image = transform_to_tensor(image=image)['image']

        return image

    @staticmethod
    def __resize_image(image: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
        image = cv2.resize(src=image, dsize=image_size)

        return image
