DATA_DIR = "/media/paul/SSD/Datasets/TeslaPilot/storage/datasets/carla"

TRAIN_TOWN_LIST = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"]
EVAL_TOWN_LIST = ["Town06", "Town07"]

IMAGE_SIZE = (224, 224)
USE_IMAGENET_NORM = True
SEQUENCE_SIZE = 4
STATE_NOISE = True

EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 1e-4
