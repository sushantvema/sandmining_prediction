# development script to test sandmining package. 

import sandmining

import argparse
import ipdb
import os
import sys


def main(arguments):

    parser = argparse.ArgumentParser()
    parser.add_argument('command', help="What step of the process to run",
                        choices=['load_observations', 'process_annotations'])

    args = parser.parse_args()

    command = args.command
    match command:
        case 'load_observations':
            sandmining.load_observations()
        case 'process_annotations':
            sandmining.process_annotations()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))