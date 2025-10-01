#!/usr/bin/env python3
import os
import psutil
import torch
from torch.utils.data import DataLoader

from config import *
from dataset import *
from cilv3d import CILv3D
from trainer import Trainer

# EXAMPLE USAGE: MODEL_PATH=checkpoints/CILv3D.pt CHECKPOINT=checkpoints/CILv3D_best.py ./train.py

MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/CILv3D/CILv3D.pt")
CHECKPOINT = os.getenv("CHECKPOINT", None)
WRITER_PATH = os.getenv("WRITER_PATH", None)

N_WORKERS = psutil.cpu_count(logical=False)
PREFETCH_FACTOR = psutil.cpu_count(logical=False) // 2
PIN_MEMORY = not EMA

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
torch.set_warn_always(False)


if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("[+] Using device:", device)

  os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
  print("\n[*] Configuration:")
  print(f"Model path: {MODEL_PATH}")
  print(f"Checkpoint path: {CHECKPOINT}")
  print(f"Epochs: {EPOCHS} - Batch size: {BATCH_SIZE} - Learning rate: {LR} - Weight decay: {WEIGHT_DECAY}")
  print(f"Number of workers: {N_WORKERS} - Prefetch factor: {PREFETCH_FACTOR}")
  print(f"EMA: {EMA} - Pin memory: {PIN_MEMORY}")
  print(f"NORMALIZE_STATES: {NORMALIZE_STATES}")
  print()

  train_set = CarlaDataset(
    base_dir=DATA_DIR,
    townslist=TRAIN_TOWN_LIST,
    image_size=IMAGE_SIZE,
    use_imagenet_norm=USE_IMAGENET_NORM,
    sequence_size=SEQUENCE_SIZE
  )
  val_set = CarlaDataset(
    base_dir=DATA_DIR,
    townslist=EVAL_TOWN_LIST,
    image_size=IMAGE_SIZE,
    use_imagenet_norm=USE_IMAGENET_NORM,
    sequence_size=SEQUENCE_SIZE
  )

  train_loader =  DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                             prefetch_factor=PREFETCH_FACTOR, num_workers=N_WORKERS, pin_memory=PIN_MEMORY)
  val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                          prefetch_factor=PREFETCH_FACTOR, num_workers=N_WORKERS, pin_memory=PIN_MEMORY)

  # torch.set_float32_matmul_precision('high')
  model = CILv3D(device=device)
  model.to(device)
  # model = torch.compile(model)

  trainer = Trainer(
    device, model, MODEL_PATH, train_loader, val_loader,
    checkpoint_path=CHECKPOINT, writer_path=WRITER_PATH, eval_epoch=True,
    save_checkpoints=True, early_stopping=True
  )
  trainer.train()
