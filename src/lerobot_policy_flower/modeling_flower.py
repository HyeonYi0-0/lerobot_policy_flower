"""
FlowerPolicy — PreTrainedPolicy wrapper for FLOWER VLA.
"""

from typing import Any

import torch
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION

from .configuration_flower import FlowerConfig
from .modeling_flower_core import FLOWERVLACore
from .transforms import RandomShiftsAug


class FlowerPolicy(PreTrainedPolicy):
    config_class = FlowerConfig
    name = "flower"

    def __init__(
        self,
        config: FlowerConfig,
        **kwargs,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = FLOWERVLACore(config)
        if config.pretrained_model_path:
            self.model._load_pretrained_weights(config.pretrained_model_path)
        # Build per-camera aug map. random_shifts_aug_pad_per_camera overrides the
        # uniform random_shifts_aug_pad for specific camera keys.
        if config.use_random_shifts_aug:
            self._random_shifts_aug = {
                key: RandomShiftsAug(
                    pad=config.random_shifts_aug_pad_per_camera.get(key, config.random_shifts_aug_pad)
                )
                for key in config.image_features
            }
        else:
            self._random_shifts_aug = {}

    def reset(self) -> None:
        """Reset action chunking counter at episode start."""
        self.model.reset()

    def get_optim_params(self) -> list[dict]:
        # bias and LayerNorm params are excluded from weight decay
        no_decay_keywords = ["bias", "LayerNorm", "layernorm", "ln", "norm"]
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(kw in name.lower() for kw in no_decay_keywords):
                    no_decay.append(param)
                else:
                    decay.append(param)
        return [
            {"params": decay, "weight_decay": self.config.optimizer_weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def predict_action_chunk(self, batch: dict, **kwargs) -> Tensor:
        """Predict a full action chunk of shape (B, chunk_size, action_dim)."""
        features = self.model.encode_observations(batch)
        noise = torch.randn(
            features["features"].shape[0],
            self.config.chunk_size,
            self.config.action_dim,
            device=features["features"].device,
        )
        return self.model.sample_actions(noise, features, inference=True)

    def select_action(self, batch: dict, **kwargs) -> Tensor:
        """
        Called at every eval step with action chunking.
        Re-generates the chunk only every n_action_steps to amortize inference cost.

        return_act_chunk=True:  returns the full cached chunk (B, chunk_size, action_dim).
        return_act_chunk=False: returns single-step action (B, action_dim) in sequence order.
        """
        if self.model.rollout_step_counter % self.config.n_action_steps == 0:
            self.model.pred_action_seq = self.predict_action_chunk(batch)

        if self.config.return_act_chunk:
            current_action = self.model.pred_action_seq  # (B, chunk_size, action_dim)
        else:
            current_action = self.model.pred_action_seq[
                :, self.model.rollout_step_counter
            ]  # (B, action_dim)

        self.model.rollout_step_counter = (
            self.model.rollout_step_counter + 1
        ) % self.config.n_action_steps
        return current_action

    def forward(
        self, batch: dict, reduction: str = "mean"
    ) -> tuple[Tensor, dict | None]:
        """
        Args:
            reduction: signature-only; rf_loss always returns mean internally.
                       Kept for future per-sample loss support.
        """
        actions = batch[ACTION]  # (B, chunk_size, action_dim)

        if actions.ndim == 4:
            actions = actions.squeeze(1)

        if self._random_shifts_aug:
            batch = dict(batch)  # shallow copy — do not mutate the original
            for key, aug in self._random_shifts_aug.items():
                if key in batch and batch[key].ndim == 4:  # (B, C, H, W)
                    batch[key] = aug(batch[key])

        features = self.model.encode_observations(batch)
        loss, losses_dict = self.model.rf_loss(features, actions)

        # TODO: apply action_is_pad mask — rf_loss uses mean() internally so padded
        # timesteps are not fully excluded yet.
        return loss, {k: float(v) for k, v in losses_dict.items()}
