"""FLOWERVLACore — pure nn.Module without PyTorch Lightning dependency."""

import functools
from typing import TYPE_CHECKING, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .configuration_flower import FlowerConfig


class FLOWERVLACore(nn.Module):
    def __init__(self, config: "FlowerConfig"):
        super().__init__()
        from lerobot.utils.import_utils import is_package_available

        if not is_package_available("transformers"):
            raise ImportError("The 'transformers' package is required. Install it with: pip install transformers")

        self.config = config

        self._init_from_config(config)

        # VLM must be initialized first to resolve hidden_dim for DiT
        self._setup_vlm(config.vlm_path, config.freeze_vision_tower, config.freeze_florence)
        hidden_dim = self.vlm.config.text_config.d_model
        self.vlm_latent_dim = hidden_dim

        self._setup_dit_components(config, hidden_dim)

        self.rollout_step_counter = 0
        self.pred_action_seq = None

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_from_config(self, config: "FlowerConfig") -> None:
        from .flower.models.utils import ActionIndex, generate_policy_prompt

        self.use_rope = config.use_rope and not config.use_nope
        self.use_nope = config.use_nope and not config.use_rope
        self.sampling_type = config.sampling_type
        self.token_dropout = config.token_dropout
        self.use_proprio = config.use_proprio
        self.action_type_adaln = config.action_type_adaln
        self.use_cross_attn = config.use_cross_attn
        self.use_causal_attention = config.use_causal_attention
        self.use_adaln_cond = config.use_adaln_cond
        # use_readout_token is only meaningful when use_adaln_cond is True
        self.use_readout_token = config.use_readout_token and config.use_adaln_cond
        self.return_act_chunk = config.return_act_chunk
        self.vlm_prompt_style = config.vlm_prompt_style
        self.format_instruction = functools.partial(
            generate_policy_prompt,
            robot_name=config.format_instruction_robot_name,
            num_arms=config.format_instruction_num_arms,
            action_space=config.format_instruction_action_space,
            prompt_style=config.format_instruction_prompt_style,
        )

        self.dit_dim = config.dit_dim
        self.n_heads = config.n_heads
        self.action_dim = config.action_dim
        self.act_window_size = config.chunk_size
        self.multistep = config.n_action_steps
        self.num_sampling_steps = config.num_sampling_steps
        self.lowdim_obs_dim = config.lowdim_obs_dim
        self.action_space_index = ActionIndex()
        self.obs_modalities = []
        self.target_modality = "actions"
        self.modality_scope = "lang"

    def _setup_vlm(self, vlm_path: str, freeze_vision_tower: bool, freeze_florence: bool) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        print(f"Loading Florence-2 from {vlm_path}")
        self.vlm = AutoModelForCausalLM.from_pretrained(vlm_path, trust_remote_code=True)

        if freeze_florence:
            for param in self.vlm.parameters():
                param.requires_grad = False
        elif not freeze_vision_tower:
            for param in self.vlm.vision_tower.parameters():
                param.requires_grad = True

        self.processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.prompt_embeds = self._create_prompt_embed("<Flow>")

        del self.vlm.language_model.model.decoder
        del self.vlm.language_model.lm_head

        self.vlm_token_dropout = nn.Dropout(self.token_dropout)

    def _setup_dit_components(self, config: "FlowerConfig", hidden_dim: int) -> None:
        from .flower.models.networks.transformers import (
            ActionSpaceEmbedderParameter,
            FlowBlock,
            FreqEmbedder,
            RmsNorm,
            SharedAdaLNController,
            TimestepEmbedder,
            ZeroEncoder,
        )
        from timm.layers.mlp import Mlp

        self.action_encoders = nn.ModuleDict()
        self.action_decoders = nn.ModuleDict()
        if self.use_proprio:
            self.proprio_encoders = nn.ModuleDict()
        self.adaln = nn.ModuleDict() if self.action_type_adaln else None

        self.cond_linear = nn.Linear(hidden_dim, self.dit_dim, bias=False)
        self.t_embedder = TimestepEmbedder(self.dit_dim)
        self.cond_norm = RmsNorm(hidden_dim)
        self.frequency_embedder = FreqEmbedder(self.dit_dim)
        self.action_space_embedder = ActionSpaceEmbedderParameter(
            self.dit_dim, max_actions=len(self.action_space_index.action_spaces)
        )

        if not self.use_rope and not self.use_nope:
            self.positional_encoding = nn.Parameter(
                torch.randn(1, self.act_window_size, self.dit_dim) * 0.1
            )

        self.dit = nn.ModuleList([
            FlowBlock(
                self.dit_dim,
                self.n_heads,
                attn_pdrop=config.attn_pdrop,
                resid_pdrop=config.resid_pdrop,
                mlp_pdrop=config.mlp_pdrop,
                use_cross_attn=self.use_cross_attn,
                use_rope=self.use_rope,
                query_seq_len=config.query_seq_len,
                rope_theta=config.rope_theta,
            )
            for _ in range(config.n_layers)
        ])

        for action_name, action_idx in self.action_space_index.action_spaces.items():
            input_dim = self.action_space_index.get_action_dim(action_idx)
            self.action_encoders[action_name] = Mlp(
                in_features=input_dim,
                hidden_features=self.dit_dim,
                out_features=self.dit_dim,
                bias=True,
            )
            self.action_decoders[action_name] = nn.Linear(self.dit_dim, input_dim)
            if self.action_type_adaln:
                self.adaln[action_name] = SharedAdaLNController(
                    self.dit_dim,
                    global_conddim=self.dit_dim,
                    use_cross_attn=self.use_cross_attn,
                )
            if self.use_proprio:
                # eef_delta and joint_single use ZeroEncoder (return zeros) by design
                if action_name in ("bimanual_nav", "dual_lerobot"):
                    self.proprio_encoders[action_name] = Mlp(
                        in_features=input_dim,
                        hidden_features=self.dit_dim,
                        out_features=self.dit_dim,
                        drop=0.2,
                    )
                else:
                    self.proprio_encoders[action_name] = ZeroEncoder(self.dit_dim)

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------

    def encode_observations(self, batch: Dict) -> Dict:
        """
        Encode a LeRobot batch into VLM features and conditioning signals.

        merged_embeds order: [task_prompt, image_features, text_embeds]
        attention_mask:      prompt=0 (masked), image=1, text=tokenizer pad mask
        """
        device = next(self.parameters()).device
        default_type = next(self.parameters()).dtype

        present_img_keys = [k for k in self.config.image_features if k in batch]
        if not present_img_keys:
            raise ValueError(
                f"No image keys found in batch. "
                f"config.image_features={list(self.config.image_features.keys())}"
            )

        first_img = batch[present_img_keys[0]]
        B = first_img.shape[0]

        all_image_features = []
        for key in present_img_keys:
            image_tensor = batch[key]
            if image_tensor.ndim == 4:              # (B, C, H, W) → (B, 1, C, H, W)
                image_tensor = image_tensor.unsqueeze(1)
            B_, T, C, H, W = image_tensor.shape
            imgs = image_tensor.view(-1, C, H, W).to(device).to(default_type)
            if H != 224 or W != 224:
                imgs = F.interpolate(imgs, size=(224, 224), mode="bilinear", align_corners=False)
            feats = self.vlm._encode_image(imgs).to(default_type)
            feats = feats.view(B_, T * feats.shape[1], -1)
            all_image_features.append(feats)

        image_features = torch.cat(all_image_features, dim=1)

        lang_text = batch["task"]
        constructed_prompts = self.construct_prompts({"lang_text": lang_text})
        text_inputs = self.tokenizer(
            constructed_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(device)
        text_embeds = self.vlm.get_input_embeddings()(text_inputs["input_ids"])
        lang_attention_mask = text_inputs["attention_mask"].to(default_type)  # (B, text_len)

        task_prompt = self.prompt_embeds.expand(B, -1, -1).to(device)
        merged_embeds = torch.cat(
            [task_prompt, image_features, text_embeds], dim=1
        )

        prompt_mask = torch.zeros(B, task_prompt.shape[1], device=device)
        vis_attention_mask = torch.ones(image_features.shape[:2], device=device)
        attention_mask = torch.cat([prompt_mask, vis_attention_mask, lang_attention_mask], dim=1)

        features = self.vlm.get_encoder()(
            inputs_embeds=merged_embeds.to(default_type),
            attention_mask=attention_mask,
        ).last_hidden_state
        features = self.vlm_token_dropout(features)

        frequency_embeds = self.frequency_embedder(
            torch.full((B,), float(self.config.frequency), device=device, dtype=default_type)
        )

        proprio = None
        if self.use_proprio:
            from lerobot.configs.types import FeatureType
            state_keys = [
                k for k, ft in self.config.input_features.items()
                if ft.type is FeatureType.STATE and k in batch
            ]
            if state_keys:
                proprio = batch[state_keys[0]].to(device).to(default_type)

        return {
            "features": features,
            "frequency_embeds": frequency_embeds,   # (B, dit_dim)
            "action_space_embeds": None,
            "action_type": torch.full(               # (B,) scalar
                (B,), self.config.action_type_index,
                device=device, dtype=torch.long,
            ),
            "proprio": proprio,
            "attention_mask": attention_mask,
        }

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def encode_proprio(
        self,
        proprio: torch.Tensor,
        action_type: torch.Tensor,
        output_shape: Tuple,
    ) -> torch.Tensor:
        batch_size = output_shape[0]
        default_dtype = next(self.parameters()).dtype
        device = proprio.device

        if not self.use_proprio:
            return torch.zeros(batch_size, self.dit_dim, device=device, dtype=default_dtype)

        encoded = torch.zeros(batch_size, self.dit_dim, device=device, dtype=default_dtype)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                encoded[mask] = self.proprio_encoders[action_name](
                    proprio[mask].to(default_dtype)
                ).to(default_dtype)

        return encoded

    def construct_prompts(self, dataset_batch: Dict) -> List[str]:
        language_instruction = dataset_batch["lang_text"]
        text_prompts = []
        for instruction in language_instruction:
            if self.vlm_prompt_style == "default":
                text_prompts.append(self.format_instruction(instruction))
            elif self.vlm_prompt_style == "feature_focused":
                text_prompts.append(
                    f"<od>{instruction}</od>"
                    f"<grounding>identify objects and spatial relationships for robotic manipulation</grounding>"
                )
            elif self.vlm_prompt_style == "state_oriented":
                text_prompts.append(
                    f"<od>{instruction}</od>"
                    f"<referring_expression_segmentation>locate objects and regions for manipulation</referring_expression_segmentation>"
                )
            else:
                raise ValueError(f"Unknown prompt style: {self.vlm_prompt_style}")
        return text_prompts

    def _create_prompt_embed(self, prompt_text: str):
        self.tokenizer.add_special_tokens({"additional_special_tokens": [prompt_text]})
        self.vlm.resize_token_embeddings(len(self.tokenizer))
        prompt_token_id = self.tokenizer.convert_tokens_to_ids(prompt_text)
        prompt_embed = nn.Parameter(
            self.vlm.get_input_embeddings()(torch.tensor(prompt_token_id)),
            requires_grad=False,
        )
        return prompt_embed.unsqueeze(0).unsqueeze(0)

    def _get_text_embeddings(self, text: List[str], device):
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(device)
        return self.vlm.get_input_embeddings()(text_inputs["input_ids"])

    def rf_loss(self, cond: Dict, actions: torch.Tensor):
        default_dtype = next(self.parameters()).dtype

        if len(actions.shape) == 4:
            actions = actions.squeeze(1)
        b = actions.size(0)
        device = actions.device
        actions = actions.to(default_dtype)

        if self.sampling_type == "pi_zero":
            alpha, beta = 1.5, 1.0
            t = torch.distributions.Beta(alpha, beta).sample((b,)).to(device)
            t = t.clamp(max=0.999)
        elif self.sampling_type == "ln":
            t = torch.sigmoid(torch.randn((b,), device=device))
            t = t.clamp(max=0.999).to(default_dtype)
        elif self.sampling_type == "uniform":
            eps = 1e-5
            t = (torch.rand(1, device=device) + torch.arange(b, device=device) / b) % (1 - eps)
            t = t.to(default_dtype)
        else:
            raise NotImplementedError(f"Sampling type {self.sampling_type} not implemented")

        texp = t.view([b] + [1] * (actions.dim() - 1))
        z1 = torch.randn_like(actions, device=device).to(default_dtype)
        zt = (1 - texp) * actions + texp * z1

        vtheta = self.dit_forward(zt, t, cond)
        diff = (z1 - actions) - vtheta
        loss = (diff ** 2).mean()

        losses_dict = {
            "diff_min": diff.min().item(),
            "diff_max": diff.max().item(),
            "diff_mean": diff.mean().item(),
            "loss": loss.item(),
        }
        return loss, losses_dict

    def sample_actions(
        self, z: torch.Tensor, cond: Dict, inference: bool = False
    ) -> torch.Tensor:
        steps = self.num_sampling_steps if inference else 5
        b = z.size(0)
        device = z.device

        dt = 1.0 / steps
        dt_tensor = torch.tensor([dt] * b, device=device).view([b] + [1] * (z.dim() - 1))

        for i in range(steps, 0, -1):
            t_val = i / steps
            t_tensor = torch.full((b,), t_val, device=device)
            vc = self.dit_forward(z, t_tensor, cond)
            z = z - dt_tensor * vc

        return z.clamp(-1, 1)

    def dit_forward(
        self, z: torch.Tensor, t: torch.Tensor, cond_dict: Dict
    ) -> torch.Tensor:
        from .flower.models.networks.transformers import stateless_norm

        default_dtype = next(self.parameters()).dtype
        B, t_seq, d = z.shape

        cond = cond_dict["features"].to(default_dtype)
        frequency_embeds = cond_dict["frequency_embeds"].to(default_dtype)
        if frequency_embeds.ndim == 3:          # guard: (B, 1, dit_dim) → (B, dit_dim)
            frequency_embeds = frequency_embeds.squeeze(1)
        action_type = cond_dict["action_type"].to(z.device)

        if self.use_proprio and cond_dict["proprio"] is not None:
            proprio = cond_dict["proprio"].to(default_dtype)
            proprio_embeds = self.encode_proprio(proprio, action_type, (B, self.dit_dim))
        else:
            proprio_embeds = torch.zeros(B, self.dit_dim, device=z.device, dtype=default_dtype)

        z, valid_dims = self.encode_actions(z, action_type)

        if not self.use_rope and not self.use_nope:
            z = z + self.positional_encoding

        t_emb = (
            stateless_norm(self.t_embedder(t))
            + stateless_norm(frequency_embeds)
            + stateless_norm(proprio_embeds)
        )
        cond = self.cond_linear(self.cond_norm(cond))

        if self.use_adaln_cond:
            vlm_token = cond[:, 0, :] if self.use_readout_token else cond.mean(dim=1)
            global_cond = vlm_token + t_emb
        else:
            global_cond = t_emb

        cx = z
        context = cond if self.use_cross_attn else None

        if not self.action_type_adaln:
            global_adaln = self.adaln(global_cond)
        else:
            global_adaln = self.action_specific_adaln(global_cond, action_type)

        cross_attn_mask = cond_dict.get("attention_mask")
        for layer in self.dit:
            cx = layer(
                cx,
                global_cond,
                context=context,
                custom_cross_attn_mask=cross_attn_mask,
                is_causal=True,
                global_adaln=global_adaln,
            )

        return self.decode_actions(cx, action_type, valid_dims)

    def encode_actions(
        self, z: torch.Tensor, action_type: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (encoded, valid_dims).
        valid_dims is a mask with 1s only in the active dimensions of each action space;
        the encoder input is sliced to adim so each head only sees its valid dims.
        """
        default_dtype = next(self.parameters()).dtype
        action_type = action_type.to(z.device)
        batch_size = z.shape[0]
        encoded = torch.zeros(
            batch_size, z.shape[1], self.dit_dim, device=z.device, dtype=default_dtype
        )
        valid_dims = torch.zeros_like(z, dtype=default_dtype)

        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                valid_dims[mask, :, :adim] = 1
                encoded[mask] = self.action_encoders[action_name](z[mask, :, :adim]).to(default_dtype)

        return encoded, valid_dims

    def decode_actions(
        self,
        z: torch.Tensor,
        action_type: torch.Tensor,
        valid_dims: torch.Tensor,
    ) -> torch.Tensor:
        default_dtype = next(self.parameters()).dtype
        batch_size = z.shape[0]
        decoded = torch.zeros(
            batch_size, z.shape[1], self.action_dim, device=z.device, dtype=default_dtype
        )

        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                pred = self.action_decoders[action_name](z[mask]).to(default_dtype)
                decoded[mask, :, :adim] = pred[..., :adim] * valid_dims[mask, :, :adim]

        return decoded

    def action_specific_adaln(
        self, global_cond: torch.Tensor, action_type: torch.Tensor
    ) -> List[torch.Tensor]:
        default_type = next(self.parameters()).dtype
        batch_size = global_cond.shape[0]
        num_chunks = 9 if self.use_cross_attn else 6
        device = global_cond.device

        mod_signals = [
            torch.zeros(batch_size, self.dit_dim, device=device, dtype=default_type)
            for _ in range(num_chunks)
        ]

        for action_idx in range(len(self.action_space_index.action_spaces)):
            mask = (action_type == action_idx)
            if mask.any():
                action_name = self.action_space_index.get_action_name(action_idx)
                action_mod = self.adaln[action_name](global_cond[mask]).to(default_type)
                for i, signal in enumerate(action_mod):
                    mod_signals[i][mask] = signal.to(default_type)

        return mod_signals

    def _load_pretrained_weights(self, pretrained_model_path: str) -> None:
        print(f"Loading pretrained weights from {pretrained_model_path}...")

        if pretrained_model_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(pretrained_model_path)
            checkpoint = {"state_dict": state_dict}
        else:
            checkpoint = torch.load(pretrained_model_path, map_location=next(self.parameters()).device)

        state_dict = checkpoint.get("state_dict", checkpoint)

        if (
            "callbacks" in checkpoint
            and "EMA" in checkpoint["callbacks"]
            and "ema_weights" in checkpoint["callbacks"]["EMA"]
        ):
            print("Found EMA weights in checkpoint, attempting to load them...")
            ema_weights_list = checkpoint["callbacks"]["EMA"]["ema_weights"]
            original_state_dict = checkpoint.get("state_dict", checkpoint)
            state_dict = {}
            ema_idx = 0
            for param_name, original_param in original_state_dict.items():
                if ema_idx < len(ema_weights_list):
                    ema_weight = ema_weights_list[ema_idx]
                    if ema_weight.shape == original_param.shape:
                        state_dict[param_name] = ema_weight
                        ema_idx += 1
                    else:
                        found_match = False
                        for temp_idx in range(ema_idx, min(ema_idx + 20, len(ema_weights_list))):
                            if ema_weights_list[temp_idx].shape == original_param.shape:
                                state_dict[param_name] = ema_weights_list[temp_idx]
                                ema_weights_list[temp_idx], ema_weights_list[ema_idx] = (
                                    ema_weights_list[ema_idx],
                                    ema_weights_list[temp_idx],
                                )
                                ema_idx += 1
                                found_match = True
                                break
                        if not found_match:
                            print(f"Warning: No matching EMA weight for {param_name}, using original")
                            state_dict[param_name] = original_param
                else:
                    print(f"Warning: Ran out of EMA weights at {param_name}, using original")
                    state_dict[param_name] = original_param
            print(f"Matched {ema_idx} EMA weights out of {len(ema_weights_list)} total")

        # Remap keys: strip "agent." prefix and align layer name differences between
        # the original Lightning checkpoint and this module's naming conventions.
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("agent.", "")
            if "vlm.language_encoder." in new_key:
                new_key = new_key.replace(
                    "vlm.language_encoder.", "vlm.language_model.model.encoder."
                )
            new_key = new_key.replace(".mlp.c_fc1.", ".mlp.fc1.")
            new_key = new_key.replace(".mlp.c_fc2.", ".mlp.fc2.")
            new_key = new_key.replace(".mlp.c_proj.", ".mlp.proj.")
            new_state_dict[new_key] = value

        # Handle shape mismatches before load_state_dict.
        # strict=False skips missing/unexpected keys but still raises on shape mismatch.
        model_state = self.state_dict()
        filtered_state_dict = {}
        skipped_keys = []
        partial_keys = []

        for key, ckpt_value in new_state_dict.items():
            if key not in model_state:
                filtered_state_dict[key] = ckpt_value  # unexpected; strict=False will ignore
                continue

            model_shape = model_state[key].shape
            ckpt_shape = ckpt_value.shape

            if model_shape == ckpt_shape:
                filtered_state_dict[key] = ckpt_value
            elif key == "action_space_embedder.action_embeddings":
                # Checkpoint has fewer action types than the current model.
                # Copy the rows that exist in the checkpoint; leave the rest random.
                ckpt_rows = ckpt_shape[0]
                model_rows = model_shape[0]
                merged = model_state[key].clone()
                merged[:ckpt_rows] = ckpt_value
                filtered_state_dict[key] = merged
                partial_keys.append(
                    f"  Partial load '{key}': {ckpt_rows}/{model_rows} rows from checkpoint"
                )
            else:
                # Shape mismatch for other keys — skip and keep random init.
                skipped_keys.append(
                    f"  Skipped '{key}': checkpoint {list(ckpt_shape)} vs model {list(model_shape)}"
                )

        missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)
        print("Pretrained weights loaded:")
        for msg in partial_keys:
            print(msg)
        for msg in skipped_keys:
            print(msg)
        if missing_keys:
            print(f"  Missing keys ({len(missing_keys)}): ")
            for key in missing_keys:
                print(f"    {key}")
        if unexpected_keys:
            print(f"  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:10]}")
        if not skipped_keys and not partial_keys and not missing_keys and not unexpected_keys:
            print("  All keys matched successfully!")

    def reset(self) -> None:
        self.rollout_step_counter = 0
        self.pred_action_seq = None
