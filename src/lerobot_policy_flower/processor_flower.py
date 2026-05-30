"""
Pre/post-processing pipeline for FLOWER policy.

Florence-2 DaViT vision tower requires float [0, 1] + ImageNet normalization;
it does not accept raw uint8 [0, 255] images.
Registered under a separate ProcessorStepRegistry name to avoid depending on
the in-tree xvla processor implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor import EnvTransition, TransitionKey
from lerobot.datasets.factory import IMAGENET_STATS
from lerobot.utils.constants import (
    OBS_IMAGES,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

if TYPE_CHECKING:
    from .configuration_flower import FlowerConfig


@dataclass
@ProcessorStepRegistry.register(name="flower_image_to_float")
class FlowerImageToFloatProcessorStep(ProcessorStep):
    """Convert image observations from uint8 [0, 255] to float32 [0, 1]."""

    image_keys: list[str] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        if not obs:
            return new_transition
        obs = obs.copy()
        keys = self.image_keys or [k for k in obs if k.startswith(OBS_IMAGES)]
        for key in keys:
            if key in obs and isinstance(obs[key], torch.Tensor):
                t = obs[key]
                obs[key] = t.float() / 255.0 if t.max() > 1.0 else t.float()
        new_transition[TransitionKey.OBSERVATION] = obs
        return new_transition

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {"image_keys": self.image_keys}


@dataclass
@ProcessorStepRegistry.register(name="flower_imagenet_normalize")
class FlowerImageNetNormalizeProcessorStep(ProcessorStep):
    """ImageNet normalization for Florence-2 DaViT. Input: float [0, 1]."""

    image_keys: list[str] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        if not obs:
            return new_transition
        obs = obs.copy()
        keys = self.image_keys or [k for k in obs if k.startswith(OBS_IMAGES)]
        for key in keys:
            if key in obs and isinstance(obs[key], torch.Tensor):
                t = obs[key]
                mean = torch.tensor(
                    IMAGENET_STATS["mean"], device=t.device, dtype=t.dtype
                ).view(3, 1, 1)
                std = torch.tensor(
                    IMAGENET_STATS["std"], device=t.device, dtype=t.dtype
                ).view(3, 1, 1)
                # broadcast to match t (BCHW or BNCHW)
                while mean.dim() < t.dim():
                    mean = mean.unsqueeze(0)
                    std = std.unsqueeze(0)
                obs[key] = (t - mean) / std
        new_transition[TransitionKey.OBSERVATION] = obs
        return new_transition

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {"image_keys": self.image_keys}


def make_flower_pre_post_processors(
    config: "FlowerConfig",
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Build pre- and post-processing pipelines for FLOWER policy.

    Preprocessing (observation → model):
      1. RenameObservationsProcessorStep        — no-op rename
      2. AddBatchDimensionProcessorStep         — add batch dim for single-sample inference
      3. FlowerImageToFloatProcessorStep        — [0, 255] → [0, 1]
      4. FlowerImageNetNormalizeProcessorStep   — ImageNet normalization for Florence-2 DaViT
      5. DeviceProcessorStep                    — move to config.device
      6. NormalizerProcessorStep                — STATE: MEAN_STD, ACTION: MEAN_STD, VISUAL: IDENTITY
         (VISUAL=IDENTITY because ImageNet normalization is already done in steps 3-4)

    Postprocessing (model output → execution):
      1. UnnormalizerProcessorStep  — unnormalize ACTION
      2. DeviceProcessorStep        — move to CPU
    """
    features = {**config.input_features, **config.output_features}

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        FlowerImageToFloatProcessorStep(),
        FlowerImageNetNormalizeProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features=features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]

    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
