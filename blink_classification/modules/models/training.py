"""Module with training code"""

from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torchmetrics
from torch.utils.data import DataLoader

from blink_classification.modules.data.utils import (
    create_data_loader,
    get_train_val_datasets,
)
from blink_classification.modules.models.network import BlinkNet


class EyeBlinkModel(pl.LightningModule):
    def __init__(
        self,
        data_folder: str,
        image_size: Tuple[int, int] = (128, 128),
        valid_percent: float = 0.3,
        in_channels: int = 3,
        num_of_output_nodes: int = 2,
        model_type: str = 'resnet18',
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        learning_rate: float = 0.001,
        num_workers: int = 6,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_type = model_type

        self.model = BlinkNet(
            model_type=model_type,
            in_channels=in_channels,
            num_of_output_nodes=num_of_output_nodes,
        )
        self.loss = torch.nn.CrossEntropyLoss()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers

        self.f1_metric = torchmetrics.F1(num_classes=2)
        self.accuracy_metric = torchmetrics.Accuracy(num_classes=2)

        self.train_dataset, self.val_dataset = get_train_val_datasets(
            data_folder=data_folder, image_size=image_size, valid_percent=valid_percent
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        result = self.model(image)

        return result

    def training_step(
        self,
        batch: Dict[str, Union[torch.Tensor, int]],
        batch_id: int,  # pylint: disable=W0613
    ) -> Dict[str, torch.Tensor]:
        image = batch['image']
        label = batch['label']

        result = self.model(image)
        loss = self.loss(result, label)

        f1_value = self.f1_metric(result, label)
        accuracy_value = self.accuracy_metric(result, label)

        self.log(name='train_loss', value=loss, prog_bar=True)
        self.log(name='train_f1', value=f1_value, prog_bar=True)
        self.log(name='train_accuracy', value=accuracy_value, prog_bar=True)

        result = {'loss': loss}

        return result

    def validation_step(
        self,
        batch: Dict[str, Union[torch.Tensor, int]],
        batch_id: int,  # pylint: disable=W0613
    ) -> Dict[str, torch.Tensor]:
        image = batch['image']
        label = batch['label']

        result = self.model(image)
        loss = self.loss(result, label)

        f1_value = self.f1_metric(result, label)
        accuracy_value = self.accuracy_metric(result, label)

        self.log(name='val_loss', value=loss, prog_bar=True)
        self.log(name='val_f1', value=f1_value, prog_bar=True)
        self.log(name='val_accuracy', value=accuracy_value, prog_bar=True)

        result = {'loss': loss}

        return result

    def test_step(
        self,
        batch: Dict[str, Union[torch.Tensor, int]],
        batch_id: int,  # pylint: disable=W0613
    ) -> Dict[str, torch.Tensor]:
        image = batch['image']
        label = batch['label']

        result = self.model(image)
        loss = self.loss(result, label)

        result = {'loss': loss}

        return result

    def training_epoch_end(
        self, outputs: Dict[str, torch.Tensor]
    ) -> None:  # pylint: disable=R0201
        pass

    def validation_epoch_end(
        self, outputs: Dict[str, torch.Tensor]
    ) -> float:  # pylint: disable=R0201
        pass

    def train_dataloader(self) -> DataLoader:
        train_dataloader = create_data_loader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        val_dataloader = create_data_loader(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return val_dataloader

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, factor=0.5, patience=3, mode='min'
        )

        configuration = {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'train_loss'},
        }

        return configuration
