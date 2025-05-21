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
      ema = False,
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
    self.ema = ema
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
    if self.ema:
      torch.save(self.ema_model.module.state_dict(), chpt_path)
    else:
      torch.save(self.model.state_dict(), chpt_path)
    print(f"[+] Checkpoint saved at {chpt_path}. New min eval loss {min_loss}")

  def train_step(self, t, step, sample_batched, loss_func, optim, epoch_losses, epoch_steer_losses, epoch_accel_losses):
    LEFT = sample_batched[0]["rgb_left"].to(self.device)
    FRONT = sample_batched[0]["rgb_front"].to(self.device)
    RIGHT = sample_batched[0]["rgb_right"].to(self.device)
    STATES = sample_batched[0]["states"].to(self.device)
    COMMANDS = sample_batched[0]["commands"].to(self.device)
    Y = sample_batched[1].to(self.device)
    out = self.model(LEFT, FRONT, RIGHT, STATES, COMMANDS)

    optim.zero_grad()
    loss = loss_func(out, Y).mean()
    steer_loss = loss_func(out[:, 0], Y[:, 0]).mean().item()
    accel_loss = loss_func(out[:, 1], Y[:, 1]).mean().item()
    loss.backward()
    optim.step()

    if self.ema:
      self.ema_model.update_parameters(self.model)

    # logging
    self.writer.add_scalar("running train loss", loss.item(), step)
    epoch_losses.append(loss.item())

    self.writer.add_scalar("steer loss", steer_loss, step)
    epoch_steer_losses.append(steer_loss)

    self.writer.add_scalar("accel loss", accel_loss, step)
    epoch_accel_losses.append(accel_loss)

    t.set_description(f"[train] Batch loss: {loss.item():.4f} - Steer loss: {steer_loss:.4f} - Accel loss: {accel_loss:.4f}")

  # TODO: early stopping
  def train(self):
    loss_func = nn.L1Loss()
    self.optim = torch.optim.AdamW(self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE, verbose=True)

    if self.ema:
      self.ema_model = torch.optim.swa_utils.AveragedModel(
        self.model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
      )

    losses, vlosses = [], []
    steer_losses, steer_vlosses = [], []
    accel_losses, accel_vlosses = [], []

    try:
      min_epoch_vloss = float("inf")
      step = 0
      vstep = 0
      stop_cnt = 0
      print("[*] Training...")
      for epoch in range(EPOCHS):
        self.model.train()
        print(f"\n[=>] Epoch {epoch+1}/{EPOCHS}")
        epoch_losses, epoch_vlosses = [], []
        epoch_steer_losses, epoch_steer_vlosses = [], []
        epoch_accel_losses, epoch_accel_vlosses = [], []

        for i_batch, sample_batched in enumerate((t := tqdm(self.train_loader))):
          self.train_step(t, step, sample_batched, loss_func, self.optim, epoch_losses, epoch_steer_losses, epoch_accel_losses)
          step += 1

        # logging
        avg_epoch_loss = np.array(epoch_losses).mean()
        losses.append(avg_epoch_loss)
        self.writer.add_scalar("epoch training loss", avg_epoch_loss, epoch)

        avg_epoch_steer_loss = np.array(epoch_steer_losses).mean()
        steer_losses.append(avg_epoch_steer_loss)
        self.writer.add_scalar("epoch steer loss", avg_epoch_steer_loss, epoch)

        avg_epoch_accel_loss = np.array(epoch_accel_losses).mean()
        accel_losses.append(avg_epoch_accel_loss)
        self.writer.add_scalar("epoch accel loss", avg_epoch_accel_loss, epoch)

        print("[->] Epoch average training loss: %.4f - steer loss: %.4f - acceleration loss: %.4f"%(avg_epoch_loss, avg_epoch_steer_loss, avg_epoch_accel_loss))

        avg_epoch_vloss = None
        if self.eval_epoch:
          vstep = self.eval(loss_func, vstep, epoch_vlosses, epoch_steer_vlosses, epoch_accel_vlosses)

          # logging
          avg_epoch_vloss = np.array(epoch_vlosses).mean()
          vlosses.append(avg_epoch_vloss)
          self.writer.add_scalar("epoch validation loss", avg_epoch_vloss, epoch)

          avg_epoch_steer_vloss = np.array(epoch_steer_vlosses).mean()
          steer_vlosses.append(avg_epoch_steer_vloss)
          self.writer.add_scalar("epoch steer validation loss", avg_epoch_steer_vloss, epoch)

          avg_epoch_accel_vloss = np.array(epoch_accel_vlosses).mean()
          accel_vlosses.append(avg_epoch_accel_vloss)
          self.writer.add_scalar("epoch accel validation loss", avg_epoch_accel_vloss, epoch)

          print("[->] Epoch average validation loss: %.4f - steer loss: %.4f - acceleration loss: %.4f"%(avg_epoch_vloss, avg_epoch_steer_vloss, avg_epoch_accel_vloss))

        # save checkpoints or early stop
        if self.save_checkpoints and avg_epoch_vloss is not None and avg_epoch_vloss < min_epoch_vloss:
          min_epoch_vloss = avg_epoch_vloss
          self.save_checkpoint(min_epoch_vloss)
        else:
          stop_cnt += 1
          if self.early_stopping and stop_cnt >= EARLY_STOP_EPOCHS:
            print(f"[!] Early stopping at epoch {epoch+1}/{EPOCHS}.")
            break
    except KeyboardInterrupt:
      print("[*] Training interrupted. Saving model...")

    print("[+] Training done")
    if self.ema:
      torch.save(self.ema_model.module.state_dict(), self.model_path)
    else:
      torch.save(self.model.state_dict(), self.model_path)
    print(f"[+] Model saved at {self.model_path}")

  def eval_step(self, t, vstep, sample_batched, loss_func, epoch_vlosses, epoch_steer_vlosses, epoch_accel_vlosses):
    LEFT = sample_batched[0]["rgb_left"].to(self.device)
    FRONT = sample_batched[0]["rgb_front"].to(self.device)
    RIGHT = sample_batched[0]["rgb_right"].to(self.device)
    STATES = sample_batched[0]["states"].to(self.device)
    COMMANDS = sample_batched[0]["commands"].to(self.device)
    Y = sample_batched[1].to(self.device)

    if self.ema:
      out = self.ema_model(LEFT, FRONT, RIGHT, STATES, COMMANDS)
    else:
      out = self.model(LEFT, FRONT, RIGHT, STATES, COMMANDS)

    loss = loss_func(out, Y).mean()
    steer_loss = loss_func(out[:, 0], Y[:, 0]).mean().item()
    accel_loss = loss_func(out[:, 1], Y[:, 1]).mean().item()

    if self.scheduler:
      self.scheduler.step(loss)
      # print(f"LR: {self.optim.param_groups[0]['lr']}")

    # logging
    self.writer.add_scalar("running val loss", loss.item(), vstep)
    epoch_vlosses.append(loss.item())

    self.writer.add_scalar("steer val loss", steer_loss, vstep)
    epoch_steer_vlosses.append(steer_loss)

    self.writer.add_scalar("accel val loss", accel_loss, vstep)
    epoch_accel_vlosses.append(accel_loss)

    t.set_description(f"[val] Batch loss: {loss.item():.4f} - Steer loss: {steer_loss:.4f} - Accel loss: {accel_loss:.4f}")

  def eval(self, loss_func, vstep, epoch_vlosses, epoch_steer_vlosses, epoch_accel_vlosses):
    with torch.no_grad():
      self.model.eval()
      try:
        for i_batch, sample_batched in enumerate((t := tqdm(self.val_loader))):
          self.eval_step(t, vstep, sample_batched, loss_func, epoch_vlosses, epoch_steer_vlosses, epoch_accel_vlosses)
          vstep += 1
      except KeyboardInterrupt:
        print("[*] Evaluation interrupted.")
    return vstep
