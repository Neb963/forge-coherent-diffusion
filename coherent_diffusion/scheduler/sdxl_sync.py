"""P03 CPU-resident, globally synchronized SDXL Euler ODE sampler.

The adapter returns CFG-combined *denoised x0* at sigma. All windows read one
frozen global latent; weighted x0 is assembled before exactly one Euler update.
No per-tile noise injection, sampler resets, global hooks or CUDA-only buffers.
This module imports torch only when loaded (existing P01/P02 remain lightweight).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True, order=True)
class Rect:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass(frozen=True)
class Tile:
    index: int
    write: Rect  # global latent coordinates
    read: Rect   # global latent coordinates, including clipped context halo


def _starts(length: int, size: int, overlap: int) -> list[int]:
    if length <= size:
        return [0]
    stride = size - overlap
    xs = list(range(0, length - size + 1, stride))
    if xs[-1] != length - size:
        xs.append(length - size)
    return xs


@dataclass(frozen=True)
class TilePlan:
    """Image-to-latent geometry is SDXL-specific; all public sizes are pixels."""
    width_px: int
    height_px: int
    scale: int
    tile_px: int
    overlap_px: int
    halo_px: int
    windows: tuple[Tile, ...]

    @property
    def width(self) -> int:
        return self.width_px // self.scale

    @property
    def height(self) -> int:
        return self.height_px // self.scale

    @classmethod
    def build(cls, width_px: int, height_px: int, tile_px: int,
              overlap_px: int = 128, halo_px: int = 64, *,
              scale: int = 8, max_windows: int = 4096) -> "TilePlan":
        values = (width_px, height_px, tile_px, overlap_px, halo_px, scale, max_windows)
        if any(type(v) is not int for v in values):
            raise ValueError("Tile geometry requires integer pixel dimensions")
        if min(width_px, height_px, tile_px, scale, max_windows) <= 0:
            raise ValueError("Positive image dimensions, tile and scale required")
        if overlap_px < 0 or halo_px < 0 or overlap_px >= tile_px:
            raise ValueError("Require 0 <= overlap < tile and halo >= 0")
        if any(n % scale for n in (width_px, height_px, tile_px, overlap_px, halo_px)):
            raise ValueError("Image/tile/overlap/halo must align to latent downscale")
        w, h = width_px // scale, height_px // scale
        size, overlap, halo = tile_px // scale, overlap_px // scale, halo_px // scale
        if size <= overlap:
            raise ValueError("Overlap consumes tile")
        xs, ys = _starts(w, size, overlap), _starts(h, size, overlap)
        if len(xs) * len(ys) > max_windows:
            raise ValueError("Tile count exceeds declared budget")
        windows = []
        for y in ys:
            for x in xs:
                core = Rect(x, y, min(x + size, w), min(y + size, h))
                read = Rect(max(0, core.x0 - halo), max(0, core.y0 - halo),
                            min(w, core.x1 + halo), min(h, core.y1 + halo))
                windows.append(Tile(len(windows), core, read))
        return cls(width_px, height_px, scale, tile_px, overlap_px, halo_px, tuple(windows))


def seeded_noise(plan: TilePlan, seed: int, *, batch: int = 1,
                 channels: int = 4, sigma: float = 1.0) -> torch.Tensor:
    """One CPU RNG draw across the whole canvas, independent of tile traversal."""
    if type(seed) is not int or type(batch) is not int or type(channels) is not int:
        raise ValueError("Integer seed, batch and channels required")
    if batch < 1 or channels != 4 or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("SDXL requires four channels and positive initial sigma")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((batch, channels, plan.height, plan.width),
                       generator=rng, dtype=torch.float32, device="cpu") * sigma


def _edge_weight(n: int, overlap: int, left: bool, right: bool) -> torch.Tensor:
    """Positive cosine taper at internal edges only; outer canvas edge stays 1."""
    a = torch.ones(n, dtype=torch.float32, device="cpu")
    k = min(overlap, n)
    if k:
        t = torch.arange(1, k + 1, device="cpu", dtype=torch.float32)
        ramp = .5 - .5 * torch.cos(math.pi * t / (k + 1))
        if left:
            a[:k] = torch.minimum(a[:k], ramp)
        if right:
            a[-k:] = torch.minimum(a[-k:], torch.flip(ramp, (0,)))
    return a


def _weights(tile: Tile, plan: TilePlan) -> torch.Tensor:
    r = tile.write
    ov = plan.overlap_px // plan.scale
    x = _edge_weight(r.width, ov, r.x0 > 0, r.x1 < plan.width)
    y = _edge_weight(r.height, ov, r.y0 > 0, r.y1 < plan.height)
    return (y[:, None] * x[None, :]).view(1, 1, r.height, r.width)


@dataclass(frozen=True)
class StepInfo:
    index: int
    sigma: float
    next_sigma: float
    windows: int


@dataclass(frozen=True)
class RunResult:
    latent: torch.Tensor  # CPU float32 [B, 4, H, W] at sigma 0
    steps: tuple[StepInfo, ...]
    cpu_workspace_estimate_bytes: int  # estimate, NOT measured peak CPU/GPU memory
    max_window_latent_bytes: int        # geometry estimate only
    plan: TilePlan                       # effective geometry, including retries


def _validate_schedule(sigmas) -> tuple[float, ...]:
    values = tuple(float(s) for s in sigmas)
    if len(values) < 2 or values[0] <= 0 or values[-1] != 0:
        raise ValueError("Expected decreasing Euler sigma schedule ending exactly at zero")
    if any(not math.isfinite(s) or s < 0 for s in values):
        raise ValueError("Nonfinite or negative sigma")
    if any(a <= b for a, b in zip(values, values[1:])):
        raise ValueError("Sigma schedule must strictly decrease")
    return values


def synchronized_euler(plan: TilePlan, latent: torch.Tensor, sigmas,
                       predict: Callable[[torch.Tensor, float, Tile], torch.Tensor],
                       *, device: str | torch.device = "cpu",
                       inference_dtype: torch.dtype = torch.float32,
                       cancelled: Callable[[], bool] | None = None,
                       progress: Callable[[StepInfo], None] | None = None,
                       cleanup: Callable[[], None] | None = None,
                       max_cpu_bytes: int = 4 * 1024 ** 3,
                       max_window_bytes: int = 512 * 1024 ** 2) -> RunResult:
    """Run one global Euler update/sigma interval. Predictor must return denoised x0.

    Only a clipped halo crop is sent to `device` at a time. CPU float32 global
    state/prediction buffers are required for finite, predictable residency.
    Caller owns model loading; `cleanup` runs even on preflight/interrupt/OOM.
    """
    try:
        if not isinstance(latent, torch.Tensor) or latent.device.type != "cpu" or latent.dtype != torch.float32:
            raise ValueError("Master latent must be a CPU torch.float32 tensor")
        if latent.ndim != 4 or latent.shape[0] < 1 or tuple(latent.shape[1:]) != (4, plan.height, plan.width):
            raise ValueError("SDXL master latent must be [B,4,latent_height,latent_width]")
        if not bool(torch.isfinite(latent).all()):
            raise ValueError("Nonfinite master latent")
        if inference_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("Unsupported inference dtype")
        if type(max_cpu_bytes) is not int or type(max_window_bytes) is not int or min(max_cpu_bytes, max_window_bytes) < 1:
            raise ValueError("Positive memory budgets required")
        schedule = _validate_schedule(sigmas)
        # Reject caller-supplied duplicate or incomplete window lists before any model work.
        expected = TilePlan.build(plan.width_px, plan.height_px, plan.tile_px,
                                  plan.overlap_px, plan.halo_px, scale=plan.scale)
        if len(plan.windows) != len(expected.windows):
            raise ValueError("Incomplete or duplicate tile plan")
        if {t for t in plan.windows} != set(expected.windows):
            raise ValueError("Tile coordinates do not match planned canvas")
        windows = sorted(plan.windows, key=lambda t: (t.write.y0, t.write.x0, t.index))
        batch = latent.shape[0]
        state_bytes = latent.numel() * 4
        workspace_bytes = state_bytes * 4 + plan.width * plan.height * 4
        max_tile_bytes = max(batch * 4 * t.read.width * t.read.height * 4 for t in windows)
        if workspace_bytes > max_cpu_bytes or max_tile_bytes > max_window_bytes:
            raise MemoryError("Configured CPU or inference tile memory budget exceeded")
        # Never mutate caller's input, including interrupted/error paths.
        state = latent.clone()
        infos = []
        for k, (sigma, sigma_next) in enumerate(zip(schedule, schedule[1:])):
            if cancelled is not None and cancelled():
                raise InterruptedError("Coherent tiled run interrupted")
            frozen = state
            numerator = torch.zeros_like(frozen, device="cpu")
            denominator = torch.zeros((1, 1, plan.height, plan.width),
                                      device="cpu", dtype=torch.float32)
            # Sorted stable traversal, independent of input window enumeration.
            for t in windows:
                if cancelled is not None and cancelled():
                    raise InterruptedError("Coherent tiled run interrupted")
                r = t.read
                q = t.write
                crop = frozen[:, :, r.y0:r.y1, r.x0:r.x1].clone().to(
                    device=device, dtype=inference_dtype)
                try:
                    with torch.inference_mode():
                        prediction = predict(crop, sigma, t)
                    if not isinstance(prediction, torch.Tensor) or tuple(prediction.shape) != tuple(crop.shape):
                        raise ValueError("Denoised tile shape mismatch")
                    out = prediction.detach().to(device="cpu", dtype=torch.float32, copy=True)
                    if not bool(torch.isfinite(out).all()):
                        raise ValueError("Nonfinite denoised tile prediction")
                finally:
                    del crop
                ox, oy = q.x0 - r.x0, q.y0 - r.y0
                core = out[:, :, oy:oy + q.height, ox:ox + q.width]
                weight = _weights(t, plan)
                numerator[:, :, q.y0:q.y1, q.x0:q.x1] += core * weight
                denominator[:, :, q.y0:q.y1, q.x0:q.x1] += weight
                del out, core, weight
            if not bool(torch.all(denominator > 0)):
                raise ValueError("Tile planner left uncovered latent coordinates")
            # Forge sampling_function_inner returns denoised x0; Euler d=(x-x0)/sigma.
            denoised = numerator / denominator
            state = frozen + (frozen - denoised) * ((sigma_next - sigma) / sigma)
            if not bool(torch.isfinite(state).all()):
                raise ValueError("Nonfinite Euler global step")
            info = StepInfo(k, sigma, sigma_next, len(windows))
            infos.append(info)
            if progress is not None:
                progress(info)
        return RunResult(state, tuple(infos), workspace_bytes, max_tile_bytes, plan)
    finally:
        if cleanup is not None:
            cleanup()
