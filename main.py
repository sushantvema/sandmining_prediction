# development script to test sandmining package. 

from sandmining.load_observations import load_observations
from sandmining.process_annotations import process_annotations
from sandmining.train_model import train_model
from sandmining.inference import evaluate_model

import argparse
import sys


def main(arguments):

    parser = argparse.ArgumentParser(description="Script description here-")
    parser.add_argument('command', type=str, help="What step of the process to run",
                        choices=['load_observations', 'process_annotations', 'train_model',
                                 'evaluate_model'])
    parser.add_argument("--model_types", nargs="+", type=str, default=[None],
                    help="List of models to train (choose from: unet, fcn, deeplabv3)",
                    choices=['unetplusplus', 'fcn', 'deeplabv3'])
    
    parser.add_argument("--predict-uniformly", action="store_true", 
                        help="Enable uniform prediction mode.")

    args = parser.parse_args()

    command = args.command
    models_to_train = args.model_types
    predict_uniformly = args.predict_uniformly

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
        case 'evaluate_model':
            evaluate_model(predict_uniformly=predict_uniformly)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))