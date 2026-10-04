"""Count-preserving target resizing; no image or on-disk data changes."""

import torch
from torch.nn import functional as F


def density_counts(density: torch.Tensor) -> torch.Tensor:
    """One count per sample; accumulate in float64 to reduce summation error."""
    return density.sum(dim=(1, 2, 3), dtype=torch.float64)


def resize_density_preserving_count(
    target: torch.Tensor, size: tuple[int, int]
) -> torch.Tensor:
    """Area-average, compensate pixel area, and preserve each sample's mass.

    For 480x640 -> 60x80, every output pixel sums an 8x8 input block.
    For non-integer ratios, per-sample correction also removes the mass error
    caused by overlapping adaptive-area bins. Empty density maps stay zero.
    """
    if target.ndim != 4 or target.shape[1] != 1 or not target.is_floating_point():
        raise ValueError("Expected floating-point density targets shaped (B, 1, H, W)")
    height, width = target.shape[-2:]
    out_height, out_width = size
    if not (0 < out_height <= height and 0 < out_width <= width):
        raise ValueError("Target size must be positive and no larger than the input")
    if not torch.isfinite(target).all() or (target < 0).any():
        raise ValueError("Ground-truth density must be finite and non-negative")
    if (height, width) == size:
        return target

    resized = F.interpolate(target, size=size, mode="area")
    resized = resized * (height * width / (out_height * out_width))
    original_counts = density_counts(target)
    resized_counts = density_counts(resized)
    if ((original_counts > 0) & (resized_counts <= 0)).any():
        raise RuntimeError("Resizing lost all density in a non-empty target")

    # No epsilon is added to the true counts: zero and tiny-count maps stay valid.
    scale = original_counts / torch.where(resized_counts > 0, resized_counts, 1.0)
    resized = resized * scale.to(resized.dtype).view(-1, 1, 1, 1)
    if not torch.allclose(density_counts(resized), original_counts, atol=1e-4, rtol=1e-6):
        raise RuntimeError("Density counts were not preserved; refusing to train")
    return resized
