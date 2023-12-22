# development script to test sandmining package. 

from sandmining.load_observations import load_observations
from sandmining.process_annotations import process_annotations
from sandmining.train_model import train_model

import argparse
import sys


def main(arguments):

    parser = argparse.ArgumentParser(description="Script description here-")
    parser.add_argument('command', type=str, help="What step of the process to run",
                        choices=['load_observations', 'process_annotations', 'train_model'])
    parser.add_argument("--model_types", nargs="+", type=str, default=[None],
                    help="List of models to train (choose from: unet, fcn, deeplabv3)",
                    choices=['unet', 'fcn', 'deeplabv3'])

    args = parser.parse_args()

    command = args.command
    models_to_train = args.model_types

    match command:
        case 'load_observations':
            load_observations()
        case 'process_annotations':
            process_annotations()
        case 'train_model':
            if None in models_to_train:
                raise ValueError("No model(s) specified for training")
            else:
                for model in models_to_train:
                    train_model(model)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))