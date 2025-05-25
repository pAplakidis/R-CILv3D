# CILv3D

A driving agent neural network that improves CIL++ architecture by adding uniformer for spatio-temporal features.

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

NOTE: change config.py and cilv3d.py/CILv3DConfig accordingly

## Inference

```bash
MODEL_PATH=<model_load_path> TOWN=<town_idx> EPISODE=<episode_idx> ./inference.py
```

