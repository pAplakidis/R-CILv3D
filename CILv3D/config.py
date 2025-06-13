BASE_DATA_DIR = "../data/carla"
DATA_DIR = "../data/carla"
# DATA_DIR = "/home/paul/Dev/TeslaPilot/storage/datasets/carla-cityscapes"

TRAIN_TOWN_LIST = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"]
EVAL_TOWN_LIST = ["Town06", "Town07"]

EMA = True
IMAGE_SIZE = (224, 224)
USE_IMAGENET_NORM = True
SEQUENCE_SIZE = 4
STATE_NOISE = True
NORMALIZE_STATES = True

EPOCHS = 100
BATCH_SIZE = 64 # TODO: max batch_size
LR = 5e-5
LR_FACTOR = 0.2
WEIGHT_DECAY = 1e-4
LR_PATIENCE = 10
EARLY_STOP_EPOCHS = 30
