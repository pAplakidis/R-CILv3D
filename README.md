# RCILv3D

The official implementation of the RCILv3D model (Sim-to-Real Autonomous Driving with Noise-Regularized Learning).

## Setup

```bash
pip3 install -r requirements.txt
```

## Train

```bash
MODEL_PATH=<path_to_save_model> ./train.py
```

Resume from checkpoint:

```bash
MODEL_PATH=<model_save_path> CHECKPOINT=<chekpoint_path> ./train.py
```

NOTE: change config.py and rcilv3d.py/CILv3DConfig accordingly

## Inference

```bash
MODEL_PATH=<model_load_path> TOWN=<town_idx> EPISODE=<episode_idx> ./inference.py
```
