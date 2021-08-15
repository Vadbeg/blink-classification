"""Class for model evaluation"""

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from blink_classification.modules.data.dataset import BlinkDataset
from blink_classification.modules.data.utils import (
    get_list_of_paths,
    get_val_transforms,
)
from blink_classification.modules.models.training import EyeBlinkModel


class ModelEvaluator:
    def __init__(
        self,
        images_folder: str,
        model: EyeBlinkModel,
        result_filepath: str = 'result.csv',
    ):
        self.list_of_paths = get_list_of_paths(
            data_folder=images_folder, image_file_template='*.jpg'
        )
        self.result_filepath = result_filepath

        self.model = model

        self.transforms = get_val_transforms()

        self.dataset = BlinkDataset(
            list_of_paths=self.list_of_paths,
            image_size=(128, 128),
            transforms=self.transforms,
            is_eval=True,
        )

    def evaluate(self) -> None:
        result_dataframe = pd.DataFrame(columns=['path', 'label'])

        for idx, curr_item in enumerate(tqdm(self.dataset, postfix=f'Evaluation...')):
            image = curr_item['image']
            image = self.__prepare_image(image=image, device=self.model.device)

            output = self.model(image)

            result = torch.argmax(output)
            result = result.detach().cpu().numpy()

            result_dataframe.loc[idx] = [self.dataset.list_of_paths[idx], result]

        self.__save_dataframe(dataframe=result_dataframe)

    @staticmethod
    def __prepare_image(image: torch.Tensor, device: torch.device) -> torch.Tensor:
        image = image.unsqueeze(0)
        image = image.to(device)

        return image

    def __save_dataframe(self, dataframe: pd.DataFrame) -> None:
        dataframe.to_csv(path_or_buf=self.result_filepath, header=False, index=False)
