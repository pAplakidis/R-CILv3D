#!/usr/bin/env python3
import torch
import torch.nn as nn
from dataclasses import dataclass

from config import *
from models.uniformer_video.uniformer import uniformer_small


@dataclass
class CILv3DConfig:
  sequence_size: int = SEQUENCE_SIZE
  state_size: int = 7
  command_size: int = 6
  transformer_heads: int = 8
  transformer_layers: int = 6
  filters_1d: int = 32
  filters_3d: int = 512
  embedding_size: int = 512
  freeze_backbone: bool = True


class CILv3D(nn.Module):
  def __init__(self, cfg: CILv3DConfig = CILv3DConfig()):
    super(CILv3D, self).__init__()

    self.sequence_size = cfg.sequence_size
    self.state_size = cfg.state_size
    self.command_size = cfg.command_size
    self.filters_1d = cfg.filters_1d
    self.filters_3d = cfg.filters_3d
    self.embedding_size = cfg.embedding_size
    self.transformer_heads = cfg.transformer_heads
    self.transformer_layers = cfg.transformer_layers
    self.freeze_backbone = cfg.freeze_backbone

    uniformer_state_dict = torch.load('models/state_dicts/uniformer_small_k400_16x8.pth', map_location='cpu')
    self.uniformer = uniformer_small()
    self.uniformer.load_state_dict(uniformer_state_dict)
    self.uniformer.head = nn.Identity()

    if self.freeze_backbone:
      for param in self.uniformer.parameters():
        param.requires_grad = False

    self.state_embedding = nn.Sequential(
      nn.Conv1d(in_channels=self.sequence_size, out_channels=self.filters_1d, kernel_size=1),
      nn.BatchNorm1d(self.filters_1d),
      nn.Flatten(),
      nn.Linear(self.filters_1d * self.state_size, self.embedding_size),
    )
    self.command_embedding = nn.Sequential(
      nn.Conv1d(in_channels=self.sequence_size, out_channels=self.filters_1d, kernel_size=1),
      nn.BatchNorm1d(self.filters_1d),
      nn.Flatten(),
      nn.Linear(self.filters_1d * self.command_size, self.embedding_size),
    )

    encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=self.transformer_heads)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.transformer_layers)

    self.linear = nn.Linear(self.embedding_size, 2)

  def forward(self, x, states, commands):
    """
    x: (B, C, T, H, W)
    states: (B, sequence_size, state_size)
    commands: (B, sequence_size, command_size)
    """

    state_emb = self.state_embedding(states)
    command_emb = self.command_embedding(commands)
    control_embedding = state_emb + command_emb

    # TODO: add positional embeddings (?)
    vision_embeddings, y = self.uniformer(x)
    # layerout = y[-1] # B, C, T, H, W
    # layerout = layerout[0].detach().cpu().permute(1, 2, 3, 0)

    z = vision_embeddings + control_embedding
    out = self.linear(self.transformer_encoder(z))
    return out


if __name__ == "__main__":
  model = CILv3D()
  print(model)

  bs = 2
  # FIXME: we have 3 images per time step
  vid = torch.randn(bs, 3, SEQUENCE_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1])  # B, C, T, H, W
  states = torch.randn(bs, SEQUENCE_SIZE, 7) # B, T, S
  commands = torch.randn(bs, SEQUENCE_SIZE, 6) # B, T, C
  out = model(vid, states, commands)
  print(out.shape)
