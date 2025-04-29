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
      eval_epoch: bool = False,
      skip_training: bool = False,
      save_checkpoints: bool = False,
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

    if not writer_path:
      today = str(datetime.now()).replace(" ", "_")
      auto_name = "-".join([model_path.split('/')[-1].split('.')[0], today, f"lr_{LR}", f"bs_{BATCH_SIZE}"])
      writer_path = "runs/" + auto_name
    print("[*] Tensorboard output path:", writer_path)
    self.writer = SummaryWriter(writer_path)

  def save_checkpoint(self, min_loss):
    chpt_path = self.model_path.split(".")[0] + f"_best.pt"
    torch.save(self.model.state_dict(), chpt_path)
    print(f"[+] Checkpoint saved at {chpt_path}. New min eval loss {min_loss}")

  def train_step(self, t, i_batch, sample_batched, loss_func, optim, epoch_losses):
    LEFT = sample_batched[0]["rgb_left"].permute(0, 2, 1, 3, 4).to(self.device)
    FRONT = sample_batched[0]["rgb_front"].permute(0, 2, 1, 3, 4).to(self.device)
    RIGHT = sample_batched[0]["rgb_right"].permute(0, 2, 1, 3, 4).to(self.device)
    STATES = sample_batched[0]["states"].to(self.device)
    COMMANDS = sample_batched[0]["commands"].to(self.device)
    Y = sample_batched[1].to(self.device)
    out = self.model(LEFT, FRONT, RIGHT, STATES, COMMANDS)

    optim.zero_grad()
    loss = loss_func(out, Y).mean()
    loss.backward()
    optim.step()

    self.writer.add_scalar("running loss", loss.item(), i_batch)
    epoch_losses.append(loss.item())
    t.set_description(f"[train] Batch loss: {loss.item():.4f}")

  def train(self):
    loss_func = nn.L1Loss()
    optim = torch.optim.AdamW(self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    losses, vlosses = [], []

    try:
      print("[*] Training...")
      min_epoch_vloss = float("inf")
      for epoch in range(EPOCHS):
        self.model.train()
        print(f"\n[=>] Epoch {epoch+1}/{EPOCHS}")
        epoch_losses, epoch_vlosses = [], []

        for i_batch, sample_batched in enumerate((t := tqdm(self.train_loader))):
          self.train_step(t, i_batch, sample_batched, loss_func, optim, epoch_losses)

        avg_epoch_loss = np.array(epoch_losses).mean()
        losses.append(avg_epoch_loss)
        self.writer.add_scalar("epoch training loss", avg_epoch_loss, epoch)
        print("[->] Epoch average training loss: %.4f"%(avg_epoch_loss))

        avg_epoch_vloss = None
        if self.eval_epoch:
          self.eval(loss_func, epoch_vlosses)
          avg_epoch_vloss = np.array(epoch_vlosses).mean()
          vlosses.append(avg_epoch_vloss)
          self.writer.add_scalar("epoch validation loss", avg_epoch_vloss, epoch)
          print("[->] Epoch average validation loss: %.4f"%(avg_epoch_vloss))

        if self.save_checkpoints and avg_epoch_vloss is not None and avg_epoch_vloss < min_epoch_vloss:
          min_epoch_vloss = avg_epoch_vloss
          self.save_checkpoint(min_epoch_vloss)
    except KeyboardInterrupt:
      print("[*] Training interrupted. Saving model...")

    print("[+] Training done")
    torch.save(self.model.state_dict(), self.model_path)
    print(f"[+] Model saved at {self.model_path}")

  def eval_step(self, t, i_batch, sample_batched, loss_func, epoch_vlosses):
    LEFT = sample_batched[0]["rgb_left"].permute(0, 2, 1, 3, 4).to(self.device)
    FRONT = sample_batched[0]["rgb_front"].permute(0, 2, 1, 3, 4).to(self.device)
    RIGHT = sample_batched[0]["rgb_right"].permute(0, 2, 1, 3, 4).to(self.device)
    STATES = sample_batched[0]["states"].to(self.device)
    COMMANDS = sample_batched[0]["commands"].to(self.device)
    Y = sample_batched[1].to(self.device)
    out = self.model(LEFT, FRONT, RIGHT, STATES, COMMANDS)

    loss = loss_func(out, Y).mean()
    self.writer.add_scalar("running loss", loss.item(), i_batch)
    epoch_vlosses.append(loss.item())
    t.set_description(f"[val] Batch loss: {loss.item():.4f}")

  def eval(self, loss_func, epoch_vlosses):
    with torch.no_grad():
      self.model.eval()
      try:
        for i_batch, sample_batched in enumerate((t := tqdm(self.val_loader))):
          self.eval_step(t, i_batch, sample_batched, loss_func, epoch_vlosses)
      except KeyboardInterrupt:
        print("[*] Evaluation interrupted.")
