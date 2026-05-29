"""
Image augmentation transforms for FLOWER policy training.
Adapted from: reference/flower_vla_calvin/flower/utils/transforms.py
Original source: https://github.com/facebookresearch/drqv2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomShiftsAug(nn.Module):
    """Randomly shift images horizontally and vertically via padding + random crop.

    Pads each side by `pad` pixels (replicate mode), then randomly crops back to
    the original size. This is equivalent to a random translation of up to `pad`
    pixels in any direction.

    Unlike TD-MPC's ratio-based variant, `pad` is specified in absolute pixels —
    matching the flower reference config (rgb_static: pad=10, rgb_gripper: pad=4).
    Non-square images (H != W) are supported; x and y shifts are scaled independently.

    Args:
        pad: Number of pixels to pad on each side before the random crop.
    """

    def __init__(self, pad: int):
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) float or uint8 tensor. Non-square images supported.
        Returns:
            (B, C, H, W) float tensor of same spatial size, randomly shifted.
        """
        x = x.float()
        n, c, h, w = x.size()

        x = F.pad(x, tuple([self.pad] * 4), "replicate")

        # Build separate normalized coordinate ranges for height and width axes.
        # eps accounts for the half-pixel offset in align_corners=False convention.
        eps_h = 1.0 / (h + 2 * self.pad)
        eps_w = 1.0 / (w + 2 * self.pad)

        # x-coords (width axis): shape (w,) — one value per output column
        arange_w = torch.linspace(
            -1.0 + eps_w, 1.0 - eps_w, w + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:w]
        # y-coords (height axis): shape (h,) — one value per output row
        arange_h = torch.linspace(
            -1.0 + eps_h, 1.0 - eps_h, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]

        # base_grid[i, j] = (x_j, y_i) — shape (h, w, 2)
        grid_x = arange_w.unsqueeze(0).repeat(h, 1).unsqueeze(2)  # (h, w, 1)
        grid_y = arange_h.unsqueeze(1).repeat(1, w).unsqueeze(2)  # (h, w, 1)
        base_grid = torch.cat([grid_x, grid_y], dim=2)            # (h, w, 2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)     # (n, h, w, 2)

        # Random pixel offsets, scaled to normalized coords per axis.
        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift_scale = torch.tensor(
            [2.0 / (w + 2 * self.pad), 2.0 / (h + 2 * self.pad)],
            device=x.device, dtype=x.dtype,
        )
        shift = shift * shift_scale  # (n, 1, 1, 2) * (2,) → (n, 1, 1, 2)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
