"""Resolution-independent global overview planner.

Only the low-resolution global denoiser pass must fit one requested
inference window. The full-size output is managed by synchronized tiles
with independently budgeted CPU-resident state.
"""
from __future__ import annotations


def anchor_dimensions(target, preferred_factor: int, *, alignment_px: int,
                      max_anchor_pixels: int | None = None):
    """Plan one global overview constrained by read-window area, not canvas area."""
    if type(preferred_factor) is not int or preferred_factor < 2:
        raise ValueError("Anchor scale factor must be at least two")
    if type(alignment_px) is not int or alignment_px <= 0:
        raise ValueError("Positive anchor alignment required")
    if max_anchor_pixels is not None and (
            type(max_anchor_pixels) is not int or max_anchor_pixels < 1):
        raise ValueError("Optional global-window pixel allowance must be positive")
    if not target.windows:
        raise ValueError("Anchor requires nonempty tile plan")
    read_area = max(t.read.width * t.read.height for t in target.windows)
    read_area *= target.scale ** 2
    if max_anchor_pixels is not None:
        read_area = min(read_area, max_anchor_pixels)
    max_factor = max(target.width_px, target.height_px) // alignment_px + 1
    for factor in range(preferred_factor, max_factor + 1):
        w = max(alignment_px,
                (target.width_px // factor // alignment_px) * alignment_px)
        h = max(alignment_px,
                (target.height_px // factor // alignment_px) * alignment_px)
        if w * h <= read_area:
            return w, h, factor
    raise MemoryError(
        "No patch-aligned global overview fits the selected inference window; "
        "increase the tile/window allowance or reduce attention patch alignment")


def expected_anchor_dimensions(target, factor: int, alignment_px: int):
    """Stable geometry reconstruction to detect tampered plan/lineage."""
    if type(factor) is not int or factor < 2:
        raise ValueError("Invalid global overview factor")
    return (max(alignment_px,
                target.width_px // factor // alignment_px * alignment_px),
            max(alignment_px,
                target.height_px // factor // alignment_px * alignment_px))
