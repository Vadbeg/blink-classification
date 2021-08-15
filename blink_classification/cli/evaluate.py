"""Module with functions for evaluation CLI"""


import argparse

import torch

from blink_classification.modules.evaluate.model_evaluator import ModelEvaluator
from blink_classification.modules.models.training import EyeBlinkModel


def get_args():
    parser = argparse.ArgumentParser(description='Script for model inference on folder')

    parser.add_argument('--folder-path', type=str, help='Path to folder with images.')
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        default='checkpoints/best_checkpoint.ckpt',
        help='Path to pytorch_lightning checkpoint',
    )
    parser.add_argument(
        '--config-path',
        type=str,
        default='configs/train_config.yaml',
        help='Path to training config. Is used for getting model characteristics',
    )
    parser.add_argument(
        '--device-type',
        type=str,
        choices=['cpu', 'cuda'],
        default='cuda',
        help='Which device type use for inference',
    )

    args = parser.parse_args()

    return args


def start_eval():
    cli_args = get_args()

    folder_path = cli_args.folder_path

    checkpoint_path = cli_args.checkpoint_path
    config_path = cli_args.config_path
    device_type = cli_args.device_type

    model = EyeBlinkModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path, hparams_file=config_path, evaluation=True
    )
    model.eval()
    model.to(torch.device(device_type))

    model_evaluator = ModelEvaluator(images_folder=folder_path, model=model)

    model_evaluator.evaluate()
