#!/usr/bin/env python3
import torch
import torch.nn as nn
from enum import Enum
from dataclasses import dataclass

from config import *
from models.RevIN import RevIN
from models.uniformer_video.uniformer import uniformer_small, uniformer_base

class UniformerVersion(Enum):
  SMALL = "small"
  BASE = "base"


@dataclass
class RCILv3DConfig:
  sequence_size = SEQUENCE_SIZE
  state_size = 7
  command_size = 6
  transformer_heads = 8
  transformer_layers = 6
  filters_1d = 32
  embedding_size = 512
  freeze_backbone = True
  transformer_dropout = 0.4
  linear_dropout = 0.4
  use_revin = False
  uniformer_version = UniformerVersion.BASE
  future_control_timesteps = FUTURE_CONTROL_TIMESTEPS


class RCILv3D(nn.Module):
  def __init__(self, device, cfg = RCILv3DConfig()):
    super().__init__()

    self.device = device
    self.sequence_size = cfg.sequence_size
    self.state_size = cfg.state_size
    self.command_size = cfg.command_size
    self.filters_1d = cfg.filters_1d
    self.embedding_size = cfg.embedding_size
    self.transformer_heads = cfg.transformer_heads
    self.transformer_layers = cfg.transformer_layers
    self.freeze_backbone = cfg.freeze_backbone
    self.transformer_dropout = cfg.transformer_dropout
    self.linear_dropout = cfg.linear_dropout
    self.use_revin = cfg.use_revin
    self.uniformer_version = cfg.uniformer_version
    self.future_control_timesteps = cfg.future_control_timesteps

    print(
      f"[*] RCILv3D configuration:\n"
      f"  sequence_size: {self.sequence_size}\n"
      f"  state_size: {self.state_size}\n"
      f"  command_size: {self.command_size}\n"
      f"  filters_1d: {self.filters_1d}\n"
      f"  embedding_size: {self.embedding_size}\n"
      f"  transformer_heads: {self.transformer_heads}\n"
      f"  transformer_layers: {self.transformer_layers}\n"
      f"  freeze_backbone: {self.freeze_backbone}\n"
      f"  transformer_dropout: {self.transformer_dropout}\n"
      f"  linear_dropout: {self.linear_dropout}\n"
      f"  use_revin: {self.use_revin}\n"
      f"  uniformer_version: {self.uniformer_version}\n"
    )

    if self.uniformer_version == UniformerVersion.SMALL:
      uniformer_state_dict = torch.load("models/state_dicts/uniformer_small_k400_16x8.pth", map_location=self.device, weights_only=False)
      self.uniformer = uniformer_small()
    elif self.uniformer_version == UniformerVersion.BASE:
      # uniformer_state_dict = torch.load("models/state_dicts/uniformer_base_k400_32x4.pth", map_location=self.device)
      uniformer_state_dict = torch.load("models/state_dicts/uniformer_base_k400_8x8.pth", map_location=self.device, weights_only=False)
      self.uniformer = uniformer_base()
    self.uniformer.load_state_dict(uniformer_state_dict)
    self.uniformer.head = nn.Identity()

    if self.freeze_backbone:
      for param in self.uniformer.parameters():
        param.requires_grad = False

    # state embeddings (optionally use RevIN for steer and acceleration)
    if self.use_revin:
      self.revin_target = RevIN(2)
      self.target_embedding = nn.Sequential(
        nn.Flatten(),
        nn.Linear(self.sequence_size * 2, self.embedding_size),
        nn.Dropout(self.linear_dropout)
      )

    if EMA:
      state_size = self.state_size - 2 if self.use_revin else self.state_size
      self.state_embedding = nn.Sequential(
        # nn.Conv1d(in_channels=self.sequence_size, out_channels=self.filters_1d, kernel_size=self.sequence_size),
        nn.Conv1d(in_channels=self.sequence_size, out_channels=self.filters_1d, kernel_size=self.sequence_size//2), # TODO: is this correct?
        nn.Flatten(),
        # nn.Linear(self.filters_1d * (state_size - self.sequence_size + 1), self.embedding_size),
        nn.Linear(128 if self.sequence_size == 8 else 192, self.embedding_size),  # TODO: dynamic input size based on state_size and sequence_size
        nn.Dropout(self.linear_dropout)
      )
    else:
      self.state_embedding = nn.Sequential(
        nn.Conv1d(in_channels=self.sequence_size, out_channels=self.filters_1d, kernel_size=self.sequence_size//2),
        nn.BatchNorm1d(self.filters_1d),
        nn.Flatten(),
        nn.Linear(self.filters_1d * (state_size - self.sequence_size + 1), self.embedding_size),
        nn.Dropout(self.linear_dropout)
      )

    # command embeddings
    if EMA:
      self.command_embedding = nn.Sequential(
        # nn.Conv1d(in_channels=self.sequence_size, out_channels=self.filters_1d, kernel_size=self.sequence_size),
        nn.Conv1d(in_channels=self.sequence_size, out_channels=self.filters_1d, kernel_size=self.sequence_size//2),  # TODO: is this correct?
        nn.Flatten(),
        # nn.Linear(self.filters_1d * (self.command_size - self.sequence_size + 1), self.embedding_size), # For Conv1d
        nn.Linear(96 if self.sequence_size == 8 else 160, self.embedding_size), # TODO: dynamic input size based on command_size and sequence_size
        nn.Dropout(self.linear_dropout)
      )
    else:
      self.command_embedding = nn.Sequential(
        nn.Conv1d(in_channels=self.sequence_size, out_channels=self.filters_1d, kernel_size=self.sequence_size//2),
        nn.BatchNorm1d(self.filters_1d),
        nn.Flatten(),
        nn.Linear(self.filters_1d * (self.command_size - self.sequence_size + 1), self.embedding_size), # For Conv1d
        nn.Dropout(self.linear_dropout)
      )

    # transformer
    self.layernorm = nn.LayerNorm(self.embedding_size)
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=self.embedding_size,
      nhead=self.transformer_heads,
      dropout=self.transformer_dropout,
      activation="gelu"
    )
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.transformer_layers)
    self.linear = nn.Linear(3 * self.embedding_size, 2 * self.future_control_timesteps) # (steer, acceleration) * future_control_timesteps

  def positional_encoding(self, batch_size: int, length: int, depth: int) -> torch.Tensor:
    assert depth % 2 == 0, "Depth must be even."
    half_depth = depth // 2

    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)       # (seq, 1)
    depths = torch.arange(half_depth, dtype=torch.float32).unsqueeze(0) / half_depth  # (1, depth/2)

    angle_rates = 1 / (10000 ** depths)                                      # (1, depth/2)
    angle_rads = positions * angle_rates                                     # (seq, depth/2)

    pos_encoding = torch.cat([torch.sin(angle_rads), torch.cos(angle_rads)], dim=-1)
    return pos_encoding.repeat(batch_size, 1, 1).to(self.device)

  def forward(
      self,
      left_img: torch.Tensor,
      front_img: torch.Tensor,
      right_img: torch.Tensor,
      states: torch.Tensor,
      commands: torch.Tensor
  ) -> torch.Tensor:
    """
    left: (B, C, T, H, W)
    front: (B, C, T, H, W)
    right: (B, C, T, H, W)
    states: (B, T, state_size)
    commands: (B, T, command_size)
    """

    # embeddings
    if self.use_revin:
      targets = states[:, :, -2:]
      target_emb = self.revin_target(targets, "norm")
      target_emb = self.target_embedding(target_emb)
      state_emb = self.state_embedding(states[:, :, :-2])
      state_emb = state_emb + target_emb
    else:
      state_emb = self.state_embedding(states)
    command_emb = self.command_embedding(commands)
    control_embedding = state_emb + command_emb
    control_embedding = control_embedding.unsqueeze(1).repeat(1, 3, 1)  # (B, 3, embedding_size)

    # vision backbone
    vision_emb_left, layerout_left = self.uniformer(left_img)
    vision_emb_front, layerout_front = self.uniformer(front_img)
    vision_emb_right, layerout_right = self.uniformer(right_img)

    # for visualization (layer out are intermediate features)
    # layerout_left = layerout_left[-1][0].detach().cpu().permute(1, 2, 3, 0)  # (T, H, W, C)
    # layerout_front = layerout_front[-1][0].detach().cpu().permute(1, 2, 3, 0)  # (T, H, W, C)
    # layerout_right = layerout_right[-1][0].detach().cpu().permute(1, 2, 3, 0)  # (T, H, W, C)
    layerout_left  = layerout_left[-1]   # (B, C, T, H, W)
    layerout_front = layerout_front[-1]
    layerout_right = layerout_right[-1]

    # embeddings fusion
    vision_embeddings = torch.cat([vision_emb_left.unsqueeze(1), vision_emb_front.unsqueeze(1), vision_emb_right.unsqueeze(1)], dim=1)  # (B, 3, 512)
    positional_embeddings = self.positional_encoding(batch_size=vision_embeddings.shape[0], length=3, depth=vision_embeddings.shape[-1])
    z = vision_embeddings + positional_embeddings + control_embedding

    # driving policy (transformer + linear)
    transformer_out = self.transformer_encoder(self.layernorm(z)).flatten(1, 2)
    linear_in = transformer_out + control_embedding.flatten(1)
    out = self.linear(linear_in).reshape(-1, self.future_control_timesteps, 2)  # (B, future_control_timesteps, 2)
    if self.use_revin: out = self.revin_target(out, "denorm")
    return out, [layerout_left, layerout_front, layerout_right]


if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = RCILv3D(device).to(device)
  print(model)

  bs = 12
  # TODO:
  # option1: pass each view through uniformer, then flatten + concatenate
  # option2: retrain from scratch with 3 * 3 channels
  vid = torch.randn(bs, 3, SEQUENCE_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(device)  # B, C, T, H, W

  states = torch.randn(bs, SEQUENCE_SIZE, 7).to(device)   # B, T, S
  commands = torch.randn(bs, SEQUENCE_SIZE, 6).to(device) # B, T, C
  out, _ = model(vid, vid, vid, states, commands)
  print(out.shape)
