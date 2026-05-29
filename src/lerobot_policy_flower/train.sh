#!/usr/bin/env bash
# =============================================================================
# train.sh — FLOWER VLA training script (single camera: observation.images.left_top)
#
# When input_features is explicitly set in train_config.yaml, lerobot-train skips
# the automatic feature population from the dataset (the `if not cfg.input_features:`
# branch in make_policy). Only observation.images.left_top is registered as VISUAL,
# so FLOWERVLACore.encode_observations() encodes only that camera.
#
# Prerequisites:
#   1. Activate the lerobot virtual environment.
#   2. Install lerobot_policy_flower (required for auto plugin discovery):
#        pip install -e lerobot_policy_flower/
#   3. Set dataset.repo_id / dataset.root in train_config.yaml to your actual paths.
#
# Usage:
#   bash lerobot_policy_flower/src/lerobot_policy_flower/train.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/train_config.yaml"

# input_features contains dot-separated dict keys that draccus cannot parse as
# CLI arguments — they must be set in train_config.yaml, not overridden here.

# accelerate launch \
#   --mixed_precision="bf16" \
#   --num_processes=1 \
#   --gradient_accumulation_steps=4 \
#   $(which lerobot-train) \
#   --config_path="${CONFIG_PATH}"

lerobot-train \
  --config_path="${CONFIG_PATH}"

# To override individual parameters via CLI, append them like:
#
#   lerobot-train \
#       config_path="${CONFIG_PATH}" \
#       --dataset.repo_id="your_org/your_dataset" \
#       --dataset.root="/path/to/dataset_parent" \
#       --policy.vlm_path="microsoft/Florence-2-large" \
#       --policy.pretrained_model_path="/path/to/flower_checkpoint.pt" \
#       --batch_size=4 \
#       --steps=100000 \
#       --output_dir="outputs/train/flower_left_top_only"
