"""Module for starting train script"""


import warnings

from pytorch_lightning.utilities.cli import LightningCLI

from blink_classification.modules.models.training import EyeBlinkModel

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    cli = LightningCLI(model_class=EyeBlinkModel, save_config_callback=None)
