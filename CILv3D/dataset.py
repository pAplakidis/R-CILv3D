#!/usr/bin/env python3
import os
from tqdm import tqdm
from PIL import Image
from typing import Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils import *
from config import *

class CarlaDataset(Dataset):
  def __init__(
    self,
    base_dir: str,
    townslist: Tuple[str],
    image_size: Tuple[int, int],
    use_imagenet_norm: bool,
    sequence_size: Optional[int] = None,
    inference = False
  ):
    super(CarlaDataset, self).__init__()

    self.base_dir = base_dir
    self.image_size = image_size
    self.use_imagenet_norm = use_imagenet_norm
    self.sequence_size = sequence_size
    self.inference = inference

    self._states_size = None
    self._state_shape = None
    self._commands_size = len(COMMANDS)

    self._imagenet_mean = IMAGENET_MEAN
    self._imagenet_std = IMAGENET_STD

    self.load_dataset(townslist, STATE_NOISE)

  @property
  def states_size(self) -> Optional[int]:
    return self._states_size

  @property
  def commands_size(self) -> Optional[int]:
    return self._commands_size

  @property
  def sequence_size(self) -> Optional[int]:
    return self._sequence_size

  @sequence_size.setter
  def sequence_size(self, value: Optional[int]):
    self._sequence_size = value

  @staticmethod
  def _construct_input_dict(
    left_images: torch.Tensor,
    front_images: torch.Tensor,
    right_images: torch.Tensor,
    states: torch.Tensor,
    commands: torch.Tensor,
    targets: torch.Tensor,
  ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    inputs = {
      "rgb_left": left_images,
      "rgb_front": front_images,
      "rgb_right": right_images,
      "states": states,
      "commands": commands
    }
    return inputs, targets

  @staticmethod
  def _construct_input_dict_inference(
    left_images: torch.Tensor,
    front_images: torch.Tensor,
    right_images: torch.Tensor,
    left_images_disp: np.ndarray,
    front_images_disp: np.ndarray,
    right_images_disp: np.ndarray,
    states: torch.Tensor,
    commands: torch.Tensor,
    targets: torch.Tensor,
  ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    inputs = {
      "rgb_left": left_images,
      "rgb_front": front_images,
      "rgb_right": right_images,
      "rgb_left_disp": left_images_disp,
      "rgb_front_disp": front_images_disp,
      "rgb_right_disp": right_images_disp,
      "states": states,
      "commands": commands
    }
    return inputs, targets
  
  def _apply_state_noise(self, states: np.ndarray) -> np.ndarray:
    noise = np.random.uniform(0.95, 1.05, size=self._state_shape)
    return states * noise

  def load_image(self, image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    transform_list = [transforms.Resize(self.image_size), transforms.ToTensor()]  # ToTensor() scales to [0, 1]

    if self.use_imagenet_norm:
      transform_list.append(transforms.Normalize(mean=self._imagenet_mean, std=self._imagenet_std))
    else:
      transform_list.append(transforms.Lambda(lambda x: x * (255.0 / 128.0) - 1.0)) # custom normalization: (x / 128.0) - 1.0

    transform = transforms.Compose(transform_list)

    if self.inference:
      return transform(image), image
    else:
      return transform(image)

  def load_dataset(self, townslist: Tuple[str], state_noise: bool):
    if self.sequence_size is None:
      return

    self.left_images = []
    self.front_images = []
    self.right_images = []
    self.states = []        # tuples of ("speed", "acceleration", "rotation_rads", "compass_rads", "gps_compass_bearing", "pedal_acceleration", "steer")
    self.commands = []      # tuples of ("command_LaneFollow", "command_Left", "command_Straight", "command_Right", "command_ChangeLaneLeft", "command_ChangeLaneRight")
    self.targets = []       # tuples of ("steer", "pedal_acceleration")

    for town in (t := tqdm(townslist)):
      town_dir = os.path.join(self.base_dir, town)
      t.set_description(f"[CarlaDataset] Loading {town}")
      for idx in os.listdir(town_dir):
        idx_dir = os.path.join(town_dir, idx)
        states_data, frame_ids = load_states(os.path.join(idx_dir, "states.csv"))

        target_columns = ["steer", "pedal_acceleration"]
        targets_data = states_data[target_columns].copy()

        if state_noise:
          states_data[["steer", "pedal_acceleration"]] = states_data[["steer", "pedal_acceleration"]].apply(lambda row: self._apply_state_noise(row.values), axis=1, result_type='broadcast')

        # states
        columns = ["speed", "acceleration", "rotation_rads", "compass_rads", "gps_compass_bearing", "pedal_acceleration", "steer"]
        for _, row in list(states_data[columns].iterrows())[:-1]:
          self.states.append(tuple(row))

        # commands
        columns = ["command_LaneFollow", "command_Left", "command_Straight", "command_Right", "command_ChangeLaneLeft", "command_ChangeLaneRight"]
        commands_data = states_data[columns]
        for _, row in list(commands_data[columns].iterrows())[:-1]:
          self.commands.append(tuple(row))

        # targets
        for _, row in list(targets_data[target_columns].iterrows())[1:]:
          self.targets.append(tuple(row))

        # images
        left_images_path = os.path.join(idx_dir, "sensors", "rgb_left")
        for image_path in sorted(os.listdir(left_images_path))[:-1]:
          self.left_images.append(os.path.join(left_images_path, image_path))

        front_images_path = os.path.join(idx_dir, "sensors", "rgb_front")
        for image_path in sorted(os.listdir(front_images_path))[:-1]:
          self.front_images.append(os.path.join(front_images_path, image_path))

        right_images_path = os.path.join(idx_dir, "sensors", "rgb_right")
        for image_path in sorted(os.listdir(right_images_path))[:-1]:
          self.right_images.append(os.path.join(right_images_path, image_path))

    # print(len(self.left_images), len(self.front_images), len(self.right_images), len(self.states), len(self.commands))
    assert len(self.left_images) == len(self.front_images) == len(self.right_images) == len(self.states) == len(self.commands)

  def __len__(self):
    return len(self.states) - SEQUENCE_SIZE

  # TODO: inference case could be cleaner (2 functions, one for train/val and one for inference)
  def __getitem__(self, idx):
    left_images = []
    front_images = []
    right_images = []
    states = []
    commands = []
    targets = []

    # for display during inference
    if self.inference:
      left_images_disp = []
      front_images_disp = []
      right_images_disp = []

    for i in range(SEQUENCE_SIZE):
      if self.inference:
        left_image, left_image_disp = self.load_image(self.left_images[idx+i])
        front_image, front_image_disp = self.load_image(self.front_images[idx+i])
        right_image, right_image_disp = self.load_image(self.right_images[idx+i])
      else:
        left_image = self.load_image(self.left_images[idx+i])
        front_image = self.load_image(self.front_images[idx+i])
        right_image = self.load_image(self.right_images[idx+i])

      mean = torch.tensor(self._imagenet_mean, dtype=left_image.dtype, device=left_image.device).view(-1, 1, 1)
      std = torch.tensor(self._imagenet_std, dtype=left_image.dtype, device=left_image.device).view(-1, 1, 1)

      if self.inference:
        left_images_disp.append(left_image_disp)
        front_images_disp.append(front_image_disp)
        right_images_disp.append(right_image_disp)

      if self.use_imagenet_norm:
        left_image = (left_image - mean) / std
        front_image = (front_image - mean) / std
        right_image = (right_image - mean) / std

      left_images.append(left_image)
      front_images.append(front_image)
      right_images.append(right_image)

    left_images = torch.stack(left_images, dim=1)
    front_images = torch.stack(front_images, dim=1)
    right_images = torch.stack(right_images, dim=1)

    if self.inference:
      left_images_disp = np.stack(left_images_disp, axis=0)
      front_images_disp = np.stack(front_images_disp, axis=0)
      right_images_disp = np.stack(right_images_disp, axis=0)

    states = torch.tensor(self.states[idx:idx+SEQUENCE_SIZE], dtype=torch.float32)
    commands = torch.tensor(self.commands[idx:idx+SEQUENCE_SIZE], dtype=torch.float32)
    targets = torch.tensor(self.targets[idx+SEQUENCE_SIZE], dtype=torch.float32)

    if self.inference:
      inputs, targets = CarlaDataset._construct_input_dict_inference(
        left_images=left_images,
        front_images=front_images,
        right_images=right_images,
        left_images_disp=left_images_disp,
        front_images_disp=front_images_disp,
        right_images_disp=right_images_disp,
        states=states,
        commands=commands,
        targets=targets
      )
    else:
      inputs, targets = CarlaDataset._construct_input_dict(
        left_images=left_images,
        front_images=front_images,
        right_images=right_images,
        states=states,
        commands=commands,
        targets=targets
      )
    return inputs, targets


if __name__ == "__main__":
  dataset = CarlaDataset(
    base_dir=DATA_DIR,
    townslist=TRAIN_TOWN_LIST,
    image_size=IMAGE_SIZE,
    use_imagenet_norm=USE_IMAGENET_NORM,
    sequence_size=SEQUENCE_SIZE,
    inference=True
  )

  for k, v in dataset[0][0].items():
    print(k, v.shape)
  print("targets", dataset[0][1].shape)
