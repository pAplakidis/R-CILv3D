#!/usr/bin/env python3
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import *
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

# Steering wheel setup
def draw_steering(ax, steer_val):
  ax.clear()
  ax.set_title("Steering Wheel")
  ax.axis("off")
  wheel = patches.Circle((0, 0), radius=1.0, fill=False, linewidth=3)
  ax.add_patch(wheel)
  angle = steer_val * 90  # Convert to degrees
  x = np.sin(np.radians(angle))
  y = np.cos(np.radians(angle))
  ax.plot([0, x], [0, y], color="red", linewidth=3)
  ax.set_xlim(-1.2, 1.2)
  ax.set_ylim(-1.2, 1.2)
  ax.set_aspect('equal')

# Pedal bar setup
def draw_pedal(ax, accel_val):
  ax.clear()
  ax.set_title("Gas / Brake Pedal")
  ax.axvline(x=0, color='black')
  color = 'green' if accel_val >= 0 else 'red'
  ax.barh(y=[0], width=[accel_val], color=color, height=0.3)
  ax.set_xlim(-1.1, 1.1)
  ax.set_yticks([])
  ax.set_xlabel("← Brake       Gas →")


# TODO: cleanup into functions
# TODO: show command and state values
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
    fig = plt.figure(figsize=(16, 10))
    
    ax_left_img   = plt.subplot2grid((3, 4), (0, 0))
    ax_front_img  = plt.subplot2grid((3, 4), (0, 1))
    ax_right_img  = plt.subplot2grid((3, 4), (0, 2))
    ax_steer      = plt.subplot2grid((3, 4), (1, 0), colspan=2)
    ax_accel      = plt.subplot2grid((3, 4), (1, 2))
    ax_steer_vis  = plt.subplot2grid((3, 4), (2, 0))
    ax_accel_vis  = plt.subplot2grid((3, 4), (2, 1))
    ax_command    = plt.subplot2grid((3, 4), (2, 2))

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
    ax_accel.set_ylim(-1.1, 1.1)
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
      CMD = data[0]["commands"].unsqueeze(0).to(device)
      Y = data[1].to(device)
      out = model(LEFT, FRONT, RIGHT, STATES, CMD)

      command = COMMANDS[torch.argmax(data[0]["commands"][-1]).item()]

      # predictions post processing
      pred = out[0].cpu().numpy()
      pred_steer = pred[0]
      pred_accel = pred[1]

      gt = Y.cpu().numpy()
      gt_steer = gt[0]
      gt_accel = gt[1]

      print(f"MODEL - Steer: {pred_steer}, Accel: {pred_accel}")
      print(f"GT - Steer: {gt_steer}, Accel: {gt_accel}")
      print(f"COMMAND: {command}")
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

      # draw controls
      draw_steering(ax_steer_vis, pred_steer)
      draw_pedal(ax_accel_vis, pred_accel)

      # draw command and state data
      ax_command.clear()
      ax_command.axis("off")
      ax_command.set_title("Command")
      ax_command.text(0.5, 0.5, command, fontsize=14, ha='center', va='center', wrap=True)

      # dynamic x-axis update
      if step > 95:
        ax_steer.set_xlim(step - 95, step + 5)
        ax_accel.set_xlim(step - 95, step + 5)

      plt.pause(0.1)
      step += 1
