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
        default='checkpoints/epoch=53-step=3185.ckpt',
        help='Path to pytorch_lightning checkpoint',
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
    device_type = cli_args.device_type

    model = EyeBlinkModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path, evaluation=True, data_foder=None
    )
    model.eval()
    model.to(torch.device(device_type))

    model_evaluator = ModelEvaluator(images_folder=folder_path, model=model)

    model_evaluator.evaluate()
