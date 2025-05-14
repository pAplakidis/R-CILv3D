#!/usr/bin/env python3
import os
import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from config import *
from dataset import CarlaDataset  
from cilv3d import CILv3D

# EXAMPLE USAGE: MODEL_PATH=checkpoints/CILv3D_new/CILv3D_e48_best.pt TOWN=1 EPISODE=8 ./inference.py
MODEL_PATH = os.getenv("MODEL_PATH")
TOWN = int(os.getenv("TOWN", 0))
EPISODE = int(os.getenv("EPISODE", 0))

TOWNS = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"]
TOWN_LEN = 10
EPISODE_LEN = 1000


# TODO: cleanup into functions
if __name__ == "__main__":
  if not MODEL_PATH:
    print("Usage: MODEL_PATH=<path_to_model> TOWN=<town_idx> EPISODE=<episode_idx> ./inference.py")
    exit(1)

  device = torch.device("cuda" if torch.cuda.is_available else "cpu")
  print("[+] Using device:", device)

  dataset = CarlaDataset(
    base_dir=DATA_DIR,
    townslist=TOWNS,
    image_size=IMAGE_SIZE,
    use_imagenet_norm=USE_IMAGENET_NORM,
    sequence_size=SEQUENCE_SIZE,
    inference=True
  )

  model = CILv3D(device=device)
  if MODEL_PATH:
    print(f"[+] Loading checkpoint from {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH))
  model.to(device)
  model.eval()
  print(f"[+] Model {MODEL_PATH} loaded successfully")

  pred_steers, pred_accels = [], []
  gt_steers, gt_accels = [], []

  with torch.no_grad():
   # initialize figure and layout (2 rows x 3 columns)
    fig = plt.figure(figsize=(15, 8))
    
    ax_left_img   = plt.subplot2grid((2, 3), (0, 0))
    ax_front_img  = plt.subplot2grid((2, 3), (0, 1))
    ax_right_img  = plt.subplot2grid((2, 3), (0, 2))
    ax_steer      = plt.subplot2grid((2, 3), (1, 0), colspan=2)
    ax_accel      = plt.subplot2grid((2, 3), (1, 2))

    # setup plots
    line_pred_steer, = ax_steer.plot([], [], 'r-', label='Predicted Steer')
    line_gt_steer,   = ax_steer.plot([], [], 'g-', label='GT Steer')
    ax_steer.set_xlim(0, 100)
    ax_steer.set_ylim(-1.2, 1.2)
    ax_steer.set_title("Steering Angle")
    ax_steer.set_ylabel("Steer")
    ax_steer.legend()

    line_pred_accel, = ax_accel.plot([], [], 'r-', label='Predicted Accel')
    line_gt_accel,   = ax_accel.plot([], [], 'g-', label='GT Accel')
    ax_accel.set_xlim(0, 100)
    ax_accel.set_ylim(-0.1, 1.1)
    ax_accel.set_title("Acceleration")
    ax_accel.set_xlabel("Frame")
    ax_accel.set_ylabel("Accel")
    ax_accel.legend()

    img_left = img_front = img_right = None

    step = 0
    start_idx = TOWN * TOWN_LEN + EPISODE * EPISODE_LEN
    for i in range(start_idx, len(dataset)):
      print(f"[*] Frame: {step} - Dataset Index: {i}")
      data = dataset[i]

      # feed data to model
      LEFT = data[0]["rgb_left"].unsqueeze(0).to(device)
      FRONT = data[0]["rgb_front"].unsqueeze(0).to(device)
      RIGHT = data[0]["rgb_right"].unsqueeze(0).to(device)
      STATES = data[0]["states"].unsqueeze(0).to(device)
      COMMANDS = data[0]["commands"].unsqueeze(0).to(device)
      Y = data[1].to(device)
      out = model(LEFT, FRONT, RIGHT, STATES, COMMANDS)

      # post processing
      pred = out[0].cpu().numpy()
      pred_steer = pred[0]
      pred_accel = pred[1]

      gt = Y.cpu().numpy()
      gt_steer = gt[0]
      gt_accel = gt[1]
      print(f"MODEL - Steer: {pred_steer}, Accel: {pred_accel}")
      print(f"GT - Steer: {gt_steer}, Accel: {gt_accel}")
      print()

      pred_steers.append(pred_steer)
      pred_accels.append(pred_accel)
      gt_steers.append(gt_steer)
      gt_accels.append(gt_accel)

      # update images display
      left_img_data  = data[0]["rgb_left_disp"][-1]
      front_img_data = data[0]["rgb_front_disp"][-1]
      right_img_data = data[0]["rgb_right_disp"][-1]

      if step == 0:
        img_left  = ax_left_img.imshow(left_img_data)
        ax_left_img.set_title("Left Camera View")
        img_front = ax_front_img.imshow(front_img_data)
        ax_front_img.set_title("Front Camera View")
        img_right = ax_right_img.imshow(right_img_data)
        ax_right_img.set_title("Right Camera View")
        for ax in [ax_left_img, ax_front_img, ax_right_img]:
          ax.axis("off")
      else:
        img_left.set_data(left_img_data)
        img_front.set_data(front_img_data)
        img_right.set_data(right_img_data)

      # update plots
      x_vals = list(range(len(pred_steers)))
      line_pred_steer.set_data(x_vals, pred_steers)
      line_gt_steer.set_data(x_vals, gt_steers)
      line_pred_accel.set_data(x_vals, pred_accels)
      line_gt_accel.set_data(x_vals, gt_accels)

      # dynamic x-axis update
      if step > 95:
        ax_steer.set_xlim(step - 95, step + 5)
        ax_accel.set_xlim(step - 95, step + 5)

      plt.pause(0.1)
      step += 1
