import os
from pathlib import Path

# Data Directory
PACKAGE_DIRECTORY = Path.cwd() / 'sandmining'
DATA_DIRECTORY = PACKAGE_DIRECTORY / 'data'

# Observation Directories
OBS_0_DIRECTORY = DATA_DIRECTORY / 'Observation0'
OBS_1_DIRECTORY = DATA_DIRECTORY / 'Observation1'
OBS_2_DIRECTORY = DATA_DIRECTORY / 'Observation2'

# Models Directory
MODELS_DIRECTORY = PACKAGE_DIRECTORY / 'models'