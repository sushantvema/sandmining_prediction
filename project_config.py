from pathlib import Path

# Data Directory
PACKAGE_DIRECTORY = Path.cwd() / 'sandmining'
DATA_DIRECTORY = PACKAGE_DIRECTORY / 'data'

# Observation Directories
OBSERVATIONS_JSON = DATA_DIRECTORY / 'observations.json'
OBS_0_DIRECTORY = DATA_DIRECTORY / 'Observation0'
OBS_1_DIRECTORY = DATA_DIRECTORY / 'Observation1'
OBS_2_DIRECTORY = DATA_DIRECTORY / 'Observation2'

# Models Directory
MODELS_DIRECTORY = PACKAGE_DIRECTORY / 'models'

# Data Preprocessing Parameters
IN_RIVER_BOUNDS_THRESHOLD = 0.85

# Model Parameters
NUM_CHANNELS = 3
NUM_CLASSES = 1

# Model Training Parameters
NUM_EPOCHS = 1
BATCH_SIZE = 4
SAMPLES_TO_TRAIN_PER_OBSERVATION = 1000