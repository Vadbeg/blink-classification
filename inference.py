"""Script for model inference"""

import warnings

from blink_classification.cli.evaluate import start_eval

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    start_eval()
