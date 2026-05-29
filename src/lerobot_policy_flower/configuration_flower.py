import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import LRSchedulerConfig

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler


@LRSchedulerConfig.register_subclass("tri_stage")
@dataclass
class TriStageLRSchedulerConfig(LRSchedulerConfig):
    """
    Three-phase LR schedule: warmup (linear) → hold (flat) → decay (cosine).

    Implemented via LambdaLR, so the optimizer's base_lr must equal the peak_lr.
    Set num_warmup_steps=None to auto-compute from phase_ratio[0] * num_training_steps.
    """

    # None allowed; computed from phase_ratio when build() is called
    num_warmup_steps: int | None = None

    init_lr_scale: float = 0.1              # warmup start LR = peak_lr * init_lr_scale
    final_lr_scale: float = 0.5             # decay end LR   = peak_lr * final_lr_scale
    phase_ratio: tuple = (0.05, 0.1, 0.85)  # (warmup, hold, decay) ratios — must sum to 1.0

    def build(self, optimizer: "Optimizer", num_training_steps: int) -> "LRScheduler":
        from torch.optim.lr_scheduler import LambdaLR

        warmup_steps = (
            self.num_warmup_steps
            if self.num_warmup_steps is not None
            else int(num_training_steps * self.phase_ratio[0])
        )
        hold_steps = int(num_training_steps * self.phase_ratio[1])
        decay_steps = num_training_steps - warmup_steps - hold_steps

        init_lr_scale = self.init_lr_scale
        final_lr_scale = self.final_lr_scale

        def lr_lambda(current_step: int) -> float:
            # Warmup: linear ramp init_lr_scale → 1.0
            if current_step < warmup_steps:
                if warmup_steps == 0:
                    return 1.0
                t = current_step / warmup_steps
                return init_lr_scale + (1.0 - init_lr_scale) * t

            step_after_warmup = current_step - warmup_steps

            # Hold: keep peak_lr
            if step_after_warmup < hold_steps:
                return 1.0

            step_in_decay = step_after_warmup - hold_steps

            # Decay: cosine ramp peak_lr → final_lr
            if step_in_decay <= decay_steps:
                progress = step_in_decay / max(1, decay_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return final_lr_scale + (1.0 - final_lr_scale) * cosine

            return final_lr_scale

        return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


@PreTrainedConfig.register_subclass("flower")
@dataclass
class FlowerPolicyConfig(PreTrainedConfig):
    # VLM settings
    vlm_path: str = "microsoft/Florence-2-large"
    freeze_florence: bool = False
    freeze_vision_tower: bool = False
    vlm_prompt_style: str = "default"
    token_dropout: float = 0.1

    # Action chunking
    chunk_size: int = 50
    n_action_steps: int = 50 # (lerobot dataset fps: 20)
    num_sampling_steps: int = 8 # ALOHA (bi-arm) uses 8 steps

    # Observation dims (6-DoF bi-arm)
    lowdim_obs_dim: int = 12
    # Action dim (6-DoF bi-arm)
    action_dim: int = 12

    # DiT architecture
    dit_dim: int = 1024
    n_heads: int = 16
    n_layers: int = 18
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    mlp_pdrop: float = 0.1
    sampling_type: str = "uniform"

    # RoPE
    use_rope: bool = True
    use_nope: bool = False
    query_seq_len: int = 100
    rope_theta: float = 32.0

    # Model flags
    action_type_adaln: bool = True
    use_causal_attention: bool = True
    use_cross_attn: bool = True
    use_adaln_cond: bool = False
    use_readout_token: bool = False
    use_proprio: bool = True # ALOHA (bi-arm) uses proprio
    return_act_chunk: bool = False

    normalization_mapping: dict = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    optimizer_lr: float = 2e-5
    optimizer_weight_decay: float = 0.05
    optimizer_betas: tuple = (0.9, 0.95)

    # Sampling frequency (Hz) passed as a scalar condition to FreqEmbedder
    frequency: int = 20

    # Action space index as defined in ActionIndex.
    # 0: joint_single (8D), 1: eef_delta (7D), 2: bimanual_nav (16D), 3: dual_lerobot (12D)
    # Must match the dataset/robot; used to fill the action_type tensor in encode_observations.
    action_type_index: int = 3  # default: dual_lerobot (12D joint, 2-arm)

    # Format instruction parameters — used in generate_policy_prompt via construct_prompts.
    # Applied when vlm_prompt_style == "default".
    format_instruction_robot_name: str = "dual lerobot"
    format_instruction_action_space: str = "12D Joint Position"
    format_instruction_num_arms: int = 2
    format_instruction_prompt_style: str = "minimal"

    # Optional pretrained FLOWER checkpoint (.pt / .ckpt / .safetensors)
    pretrained_model_path: str | None = None

    # Training-time image augmentation — RandomShiftsAug (pad-and-random-crop).
    # Applied only in forward() (training), not during select_action() (inference).
    # pad is in absolute pixels; reference values: rgb_static=10, rgb_gripper=4.
    # random_shifts_aug_pad_per_camera overrides random_shifts_aug_pad per camera key.
    # example: {"observation.images.rgb_static": 10, "observation.images.rgb_gripper": 4}
    use_random_shifts_aug: bool = True
    random_shifts_aug_pad: int = 10
    random_shifts_aug_pad_per_camera: dict = field(default_factory=dict)

    scheduler_init_lr_scale: float = 0.1
    scheduler_final_lr_scale: float = 0.5
    scheduler_phase_ratio: tuple = (0.05, 0.1, 0.85)

    def __post_init__(self):
        super().__post_init__()
        if self.chunk_size <= 0:
            raise ValueError("`chunk_size` must be strictly positive.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"`n_action_steps` ({self.n_action_steps}) must be <= `chunk_size` ({self.chunk_size})."
            )

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("FlowerPolicy requires at least one VISUAL feature in input_features.")
        if self.action_feature is None:
            raise ValueError("FlowerPolicy requires 'action' in output_features.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            betas=self.optimizer_betas,
        )

    def get_scheduler_preset(self) -> TriStageLRSchedulerConfig:
        return TriStageLRSchedulerConfig(
            num_warmup_steps=None,
            init_lr_scale=self.scheduler_init_lr_scale,
            final_lr_scale=self.scheduler_final_lr_scale,
            phase_ratio=self.scheduler_phase_ratio,
        )

    @property
    def observation_delta_indices(self) -> list[int] | None:
        # Single-frame input: FLOWER encode_observations uses only the current frame
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
