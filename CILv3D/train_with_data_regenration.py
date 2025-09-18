#!/usr/bin/env python3
import os
import sys
import subprocess

PYTHON = sys.executable

# === Configuration ===
MODEL_PATH = "checkpoints/sim2real_artifacts_val/CILv3D_artifacts_val.pt"       # save
CHECKPOINT_PATH = MODEL_PATH  # load
WRITER_PATH = "runs/CILv3D_sim2real_artifacts_val"

# Path to your scripts
TRAIN_SCRIPT = "train.py"
CREATE_DATASET_SCRIPT = "D:\\Projects\\DPGAN\\dpgan\\create_sim2real_dataset.py"

# How many cycles of (train -> regenerate dataset) to run
N_CYCLES = 30


def run_cmd(cmd, env=None):
    """Run a shell command and stream its output."""
    print(f"\n[RUN] {' '.join(cmd)}\n")
    subprocess.run(cmd, env=env, check=True, stdout=None, stderr=None)


def main():
    env = os.environ.copy()
    env["MODEL_PATH"] = MODEL_PATH
    env["WRITER_PATH"] = WRITER_PATH

    # === First training run (no checkpoint) ===
    env.pop("CHECKPOINT", None)   # remove checkpoint for the first run
    run_cmd([PYTHON, TRAIN_SCRIPT], env=env)

    # === Loop for regeneration + resume training ===
    for cycle in range(1, N_CYCLES + 1):
        print(f"\n[***] Cycle {cycle}/{N_CYCLES} [***]\n")

        # 1. Regenerate dataset
        run_cmd([PYTHON, CREATE_DATASET_SCRIPT], env=env)

        # 2. Resume training from checkpoint
        env["CHECKPOINT"] = CHECKPOINT_PATH
        run_cmd([PYTHON, TRAIN_SCRIPT], env=env)


if __name__ == "__main__":
    main()
