#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
from dataclasses import dataclass

from config import *
from models.RevIN import RevIN
from models.uniformer_video.uniformer import uniformer_small

EMA = bool(os.getenv("EMA", False))


@dataclass
class CILv3DConfig:
  sequence_size: int = SEQUENCE_SIZE
  state_size: int = 7
  command_size: int = 6
  transformer_heads: int = 8
  transformer_layers: int = 6
  filters_1d: int = 32
  embedding_size: int = 512
  freeze_backbone: bool = True
  transformer_dropout: float = 0.4
  use_revin: bool = True


class CILv3D(nn.Module):
  def __init__(self, device, cfg: CILv3DConfig = CILv3DConfig()):
    super(CILv3D, self).__init__()

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
    self.use_revin = cfg.use_revin

    uniformer_state_dict = torch.load('models/state_dicts/uniformer_small_k400_16x8.pth', map_location='cpu')
    self.uniformer = uniformer_small()
    self.uniformer.load_state_dict(uniformer_state_dict)
    self.uniformer.head = nn.Identity()

    if self.freeze_backbone:
      for param in self.uniformer.parameters():
        param.requires_grad = False

    # state embeddings
    if self.use_revin:
      self.revin_state = RevIN(self.state_size)
      self.state_embedding = nn.Sequential(
        nn.Flatten(),
        nn.Linear(self.sequence_size * self.state_size, self.embedding_size),
        nn.Dropout(self.transformer_dropout)
      )
    else:
      if EMA:
        self.state_embedding = nn.Sequential(
          nn.Conv1d(in_channels=self.sequence_size, out_channels=self.filters_1d, kernel_size=self.sequence_size),
          nn.BatchNorm1d(self.filters_1d),
          nn.Flatten(),
          nn.Linear(self.filters_1d * (self.state_size - self.sequence_size + 1), self.embedding_size),
          nn.Dropout(self.transformer_dropout)
        )
      else:
        self.state_embedding = nn.Sequential(
          nn.Conv1d(in_channels=self.sequence_size, out_channels=self.filters_1d, kernel_size=self.sequence_size),
          nn.Flatten(),
          nn.Linear(self.filters_1d * (self.state_size - self.sequence_size + 1), self.embedding_size),
          nn.Dropout(self.transformer_dropout)
        )

    # command embeddings
    if EMA:
      self.command_embedding = nn.Sequential(
        nn.Conv1d(in_channels=self.sequence_size, out_channels=self.filters_1d, kernel_size=self.sequence_size),
        nn.Flatten(),
        nn.Linear(self.filters_1d * (self.command_size - self.sequence_size + 1), self.embedding_size), # For Conv1d
        nn.Dropout(self.transformer_dropout)
      )
    else:
      self.command_embedding = nn.Sequential(
        nn.Conv1d(in_channels=self.sequence_size, out_channels=self.filters_1d, kernel_size=self.sequence_size),
        nn.BatchNorm1d(self.filters_1d),
        nn.Flatten(),
        nn.Linear(self.filters_1d * (self.command_size - self.sequence_size + 1), self.embedding_size), # For Conv1d
        nn.Dropout(self.transformer_dropout)
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
    # self.linear = nn.Linear(3 * self.embedding_size, 2)
    self.linear = nn.Linear(3 * self.embedding_size, self.state_size)

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
    states: (B, sequence_size, state_size)
    commands: (B, sequence_size, command_size)
    """

    state_emb = self.state_embedding(self.revin_state(states, "norm"))
    command_emb = self.command_embedding(commands)
    control_embedding = state_emb + command_emb
    control_embedding = control_embedding.unsqueeze(1).repeat(1, 3, 1)  # (B, 3, embedding_size)

    vision_emb_left, _ = self.uniformer(left_img)
    vision_emb_front, _ = self.uniformer(front_img)
    vision_emb_right, _ = self.uniformer(right_img)

    # NOTE: uniformer returns a tuple (features, layerout), where:
    # layerout = y[-1] # B, C, T, H, W
    # layerout = layerout[0].detach().cpu().permute(1, 2, 3, 0)

    vision_embeddings = torch.cat([vision_emb_left.unsqueeze(1), vision_emb_front.unsqueeze(1), vision_emb_right.unsqueeze(1)], dim=1)  # (B, 3, 512)
    positional_embeddings = self.positional_encoding(batch_size=vision_embeddings.shape[0], length=3, depth=vision_embeddings.shape[-1])
    z = vision_embeddings + positional_embeddings + control_embedding

    transformer_out = self.transformer_encoder(self.layernorm(z)).flatten(1, 2)
    linear_in = transformer_out + command_emb.repeat(1, 3)
    out = self.linear(linear_in)
    out = self.revin_state(out, "denorm")
    return out[:, -2:]


if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = CILv3D(device).to(device)
  print(model)

  bs = 12
  # TODO:
  # option1: pass each view through uniformer, then flatten + concatenate
  # option2: retrain from scratch with 3 * 3 channels
  vid = torch.randn(bs, 3, SEQUENCE_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(device)  # B, C, T, H, W

  states = torch.randn(bs, SEQUENCE_SIZE, 7).to(device)   # B, T, S
  commands = torch.randn(bs, SEQUENCE_SIZE, 6).to(device) # B, T, C
  out = model(vid, vid, vid, states, commands)
  print(out.shape)
