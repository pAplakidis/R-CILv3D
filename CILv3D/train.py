#!/usr/bin/env python3
import os
import psutil
import torch
from torch.utils.data import DataLoader

from config import *
from dataset import *
from cilv3d import CILv3D
from trainer import Trainer

N_WORKERS = PREFETCH_FACTOR = psutil.cpu_count(logical=False)

MODEL_PATH = os.getenv("DEBUG", "checkpoints/CILv3D/CILv3D.pt")


if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available else "cpu")
  print("[+] Using device:", device)

  os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

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
                             prefetch_factor=PREFETCH_FACTOR, num_workers=N_WORKERS, pin_memory=True)
  val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                          prefetch_factor=PREFETCH_FACTOR, num_workers=N_WORKERS, pin_memory=True)

  torch.set_float32_matmul_precision('high')
  model = CILv3D(device=device)
  model.to(device)
  model = torch.compile(model)

  trainer = Trainer(device, model, MODEL_PATH, train_loader, val_loader,
                    eval_epoch=True, save_checkpoints=True)
  trainer.train()
