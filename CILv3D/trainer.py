import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import *
from lib.knee_lr_schedule import KneeLRScheduler

class Trainer:
  def __init__(
    self,
    device: torch.device,
    model,
    model_path: str,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    writer_path: Optional[str] = None,
    eval_epoch = False,
    skip_training = False,
    save_checkpoints = False,
    early_stopping = True
  ):
    self.device = device
    self.model = model
    self.model_path = model_path
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.writer_path = writer_path
    self.eval_epoch = eval_epoch
    self.skip_training = skip_training
    self.save_checkpoints = save_checkpoints
    self.scheduler = None
    self.ema_model = None
    self.early_stopping = early_stopping

    if not writer_path:
      today = str(datetime.now()).replace(" ", "_")
      auto_name = "-".join([model_path.split('/')[-1].split('.')[0], today, f"lr_{LR}", f"bs_{BATCH_SIZE}"])
      writer_path = str("runs/" + auto_name).replace(":", "_").replace(".", "_")
    print("[*] Tensorboard output path:", writer_path)
    self.writer = SummaryWriter(writer_path)

  def save_checkpoint(self, min_loss):
    chpt_path = self.model_path.split(".")[0] + f"_best.pt"
    if EMA:
      torch.save(self.ema_model.module.state_dict(), chpt_path)
    else:
      torch.save(self.model.state_dict(), chpt_path)

    if self.scheduler:
      torch.save(self.scheduler.state_dict(), chpt_path.replace(".pt", f"_scheduler.pt"))

    print(f"[+] Checkpoint saved at {chpt_path}. New min eval loss {min_loss}")

  def log_scalars(
      self,
      tag_prefix: str,
      metrics: dict,
      step: int,
      accumulators: Optional[dict] = None
  ):
    for name, value in metrics.items():
      self.writer.add_scalar(f"{tag_prefix}/{name}", value, step)
      if accumulators is not None and name in accumulators:
        accumulators[name].append(value)

  def train_step(self, t, step, sample_batched, loss_func, optim):
    LEFT = sample_batched[0]["rgb_left"].to(self.device)
    FRONT = sample_batched[0]["rgb_front"].to(self.device)
    RIGHT = sample_batched[0]["rgb_right"].to(self.device)
    STATES = sample_batched[0]["states"].to(self.device)
    COMMANDS = sample_batched[0]["commands"].to(self.device)
    WS = sample_batched[0]["steer_weight"].to(self.device)
    WA = sample_batched[0]["accel_weight"].to(self.device)
    Y = sample_batched[1].to(self.device)

    out, _ = self.model(LEFT, FRONT, RIGHT, STATES, COMMANDS)
    optim.zero_grad()

    steer_loss = loss_func(out[:, :, 0], Y[:, :, 0])
    accel_loss = loss_func(out[:, :, 1], Y[:, :, 1])
    loss = (WS * steer_loss + WA * accel_loss).mean()

    first_step_mae = loss_func(out[:, 0], Y[:, 0]).mean().item()
    mae = loss_func(out, Y).mean().item()
    steer_loss = steer_loss.mean().item()
    accel_loss = accel_loss.mean().item()

    loss.backward()
    optim.step()
    # if self.scheduler: self.scheduler.step()

    if EMA:
      self.ema_model.update_parameters(self.model)

    current_metrics = {
      "loss": loss.item(),
      "1st step mae": first_step_mae,
      "mae": mae,
      "steer loss": steer_loss,
      "accel loss": accel_loss
    }
    self.log_scalars(
      "running train",
      current_metrics,
      step,
      self.epoch_train_metrics
    )
    t.set_description("[train] " + " | ".join(
      f"{name}: {value:.4f}" for name, value in current_metrics.items()
    ))

  def train(self):
    loss_func = nn.L1Loss(reduction='none')
    self.optim = torch.optim.AdamW(self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE, verbose=True)
    # self.scheduler = KneeLRScheduler(self.optim, peak_lr=LR)

    if EMA:
      self.ema_model = torch.optim.swa_utils.AveragedModel(
        self.model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
      )

    self.train_metrics = {
      "loss": [],
      "1st step mae": [],
      "mae": [],
      "steer loss": [],
      "accel loss": []
    }
    self.val_metrics = {
      "loss": [],
      "1st step mae": [],
      "mae": [],
      "steer loss": [],
      "accel loss": []
    }

    try:
      min_epoch_vloss = float("inf")
      step = 0
      vstep = 0
      stop_cnt = 0

      print("[*] Training...")
      for epoch in range(EPOCHS):
        self.epoch_train_metrics = {
          "loss": [],
          "1st step mae": [],
          "mae": [],
          "steer loss": [],
          "accel loss": [],
        }
        self.epoch_val_metrics = {
          "loss": [],
          "1st step mae": [],
          "mae": [],
          "steer loss": [],
          "accel loss": []
        }

        self.model.train()
        print(f"\n[=>] Epoch {epoch+1}/{EPOCHS}")
        for i_batch, sample_batched in enumerate((t := tqdm(self.train_loader))):
          self.train_step(t, step, sample_batched, loss_func, self.optim)
          step += 1

        avg_metrics = {name: np.mean(values) for name, values in self.epoch_train_metrics.items()}
        self.log_scalars("epoch training", avg_metrics, epoch, self.train_metrics)
        print("[->] Epoch average training metrics: " + " | ".join(
          f"{name}: {value:.4f}" for name, value in avg_metrics.items()
        ))

        avg_epoch_vloss = None
        if self.eval_epoch:
          vstep, avg_epoch_vloss = self.eval(loss_func, vstep, epoch)

        if self.scheduler:
          self.scheduler.step(avg_epoch_vloss)
          # print(f"LR: {self.optim.param_groups[0]['lr']}")

        # save checkpoints or early stop
        if self.save_checkpoints and avg_epoch_vloss is not None and avg_epoch_vloss < min_epoch_vloss:
          min_epoch_vloss = avg_epoch_vloss # TODO: use mae instead of loss (?)
          self.save_checkpoint(min_epoch_vloss)
          stop_cnt = 0
        else:
          stop_cnt += 1
          if self.early_stopping and stop_cnt >= EARLY_STOP_EPOCHS:
            print(f"[!] Early stopping at epoch {epoch+1}/{EPOCHS}.")
            break
    except KeyboardInterrupt:
      print("[*] Training interrupted. Saving model...")

    print("[+] Training done")
    if EMA:
      torch.save(self.ema_model.module.state_dict(), self.model_path)
    else:
      torch.save(self.model.state_dict(), self.model_path)
    print(f"[+] Model saved at {self.model_path}")

    if self.scheduler:
      torch.save(self.scheduler.state_dict(), self.model_path.replace(".pt", f"_scheduler.pt"))

  def eval_step(self, t, vstep, sample_batched, loss_func):
    LEFT = sample_batched[0]["rgb_left"].to(self.device)
    FRONT = sample_batched[0]["rgb_front"].to(self.device)
    RIGHT = sample_batched[0]["rgb_right"].to(self.device)
    STATES = sample_batched[0]["states"].to(self.device)
    COMMANDS = sample_batched[0]["commands"].to(self.device)
    WS = sample_batched[0]["steer_weight"].to(self.device)
    WA = sample_batched[0]["accel_weight"].to(self.device)
    Y = sample_batched[1].to(self.device)

    if EMA:
      out, _ = self.ema_model(LEFT, FRONT, RIGHT, STATES, COMMANDS)
    else:
      out, _ = self.model(LEFT, FRONT, RIGHT, STATES, COMMANDS)

    # loss = loss_func(out, Y).mean() # TODO: calculate without MaxAbsScaler once it is implemented
    steer_loss = loss_func(out[:, :, 0], Y[:, :, 0])
    accel_loss = loss_func(out[:, :, 1], Y[:, :, 1])
    loss = (WS * steer_loss + WA * accel_loss).mean()

    first_step_mae = loss_func(out[:, 0], Y[:, 0]).mean().item()
    mae = loss_func(out, Y).mean().item()
    steer_loss = steer_loss.mean().item()
    accel_loss = accel_loss.mean().item()

    current_metrics = {
      "loss": loss.item(),
      "1st step mae": first_step_mae,
      "mae": mae,
      "steer loss": steer_loss,
      "accel loss": accel_loss
    }
    self.log_scalars(
      "running val",
      current_metrics,
      vstep,
      self.epoch_val_metrics
    )
    t.set_description("[val] " + " | ".join(
      f"{name}: {value:.4f}" for name, value in current_metrics.items()
    ))

  def eval(self, loss_func, vstep, epoch):
    with torch.no_grad():
      self.model.eval()
      for i_batch, sample_batched in enumerate((t := tqdm(self.val_loader))):
        self.eval_step(t, vstep, sample_batched, loss_func)
        vstep += 1

      avg_metrics = {name: np.mean(values) for name, values in self.epoch_val_metrics.items()}
      self.log_scalars("epoch validation", avg_metrics, epoch, self.val_metrics)
      print("[->] Epoch average validation metrics: " + " | ".join(
        f"{name}: {value:.4f}" for name, value in avg_metrics.items()
      ))

    return vstep, avg_metrics["loss"] if "loss" in avg_metrics else None
