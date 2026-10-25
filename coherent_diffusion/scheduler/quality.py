"""P04 SDXL quality execution: global-first anchor/refinement or direct synchronized tiles.

Opt-in offline API. All predictions use the P03 frozen-state global Euler scheduler;
this does not install a Forge generation callback, change model loaders or assert that
real SDXL image quality/VRAM have been validated. Experimental global-interaction
methods remain disabled pending measured image comparisons in P07.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from .sdxl_sync import (
    RunResult, StepInfo, Tile, TilePlan, seeded_noise, synchronized_euler,
    _validate_schedule,
)


@dataclass(frozen=True)
class QualityPlan:
    mode: str
    target: TilePlan
    refine_sigmas: tuple[float, ...]
    anchor: TilePlan | None
    anchor_sigmas: tuple[float, ...] | None
    anchor_factor: int
    global_interaction: str

    @classmethod
    def build(cls, target: TilePlan, *, mode: str, refine_sigmas,
              anchor_sigmas=None, anchor_factor: int = 2,
              max_anchor_pixels: int = 1_048_576,
              global_interaction: str = "none") -> "QualityPlan":
        if mode not in ("direct", "progressive"):
            raise ValueError("Only direct and progressive SDXL modes are implemented")
        if global_interaction != "none":
            raise ValueError("Global interaction experiments are disabled until validated")
        if type(anchor_factor) is not int or anchor_factor not in (2, 4):
            raise ValueError("Anchor factor must be 2 or 4")
        if type(max_anchor_pixels) is not int or max_anchor_pixels < 1:
            raise ValueError("Positive bounded anchor pixel budget required")
        refine = _validate_schedule(refine_sigmas)
        if mode == "direct":
            if anchor_sigmas is not None:
                raise ValueError("Direct mode must not silently run an anchor")
            return cls(mode, target, refine, None, None, anchor_factor, global_interaction)
        if anchor_sigmas is None:
            raise ValueError("Progressive mode requires a separate anchor sigma schedule")
        anchor_schedule = _validate_schedule(anchor_sigmas)
        divisor = target.scale * anchor_factor
        if target.width_px % divisor or target.height_px % divisor:
            raise ValueError("Anchor dimensions must align exactly; implicit crop forbidden")
        w = target.width_px // anchor_factor
        h = target.height_px // anchor_factor
        if w * h > max_anchor_pixels:
            raise MemoryError("Anchor exceeds configured global-pass pixel budget")
        # Entire anchor is evaluated as a single globally contextual work window.
        # An anchor too large for a forward pass is rejected by memory preflight.
        anchor = TilePlan.build(w, h, max(w, h), 0, 0, scale=target.scale)
        if len(anchor.windows) != 1:
            raise AssertionError("Anchor must have a single global window")
        return cls(mode, target, refine, anchor, anchor_schedule,
                   anchor_factor, global_interaction)


@dataclass(frozen=True)
class QualityStage:
    name: str
    geometry: TilePlan
    steps: tuple[StepInfo, ...]
    estimated_cpu_workspace_bytes: int
    estimated_max_window_bytes: int


@dataclass(frozen=True)
class QualityResult:
    latent: object  # CPU torch.float32 at sigma zero
    mode: str
    seed: int
    stages: tuple[QualityStage, ...]
    # This is an algorithm trace, NOT a real quality, speed or VRAM measurement.
    model_forwards: int


def _preflight(p: TilePlan, *, max_cpu_bytes: int, max_window_bytes: int) -> None:
    # P03's actual global workspace and read-window checks, before either
    # stage's noise allocation / model callback.
    state = 4 * 4 * p.width * p.height
    cpu = state * 4 + p.width * p.height * 4
    window = max(4 * t.read.width * t.read.height * 4 for t in p.windows)
    if cpu > max_cpu_bytes or window > max_window_bytes:
        raise MemoryError("Stage exceeds configured global or window memory budget")


def execute_quality(
    quality: QualityPlan, seed: int,
    predict: Callable[[object, float, Tile, str], object], *,
    device="cpu", inference_dtype=None, cancelled=None, progress=None,
    cleanup=None, max_cpu_bytes: int = 4 * 1024**3,
    max_window_bytes: int = 512 * 1024**2,
) -> QualityResult:
    """Run one coherent SDXL trajectory (two stages for progressive mode).

    The caller's predictor must return denoised x0, not epsilon/velocity.
    For progressive: generate global low-res x0, bilinearly lift it, add one
    seeded target-wide Gaussian field at the *refinement starting sigma*, then
    synchronize the refinement trajectory on the target canvas. This is not
    independent tile txt2img or simple post-generation pixel upscaling.

    A predictor may be a prepared Forge adapter, but stage-sized SDXL
    conditioning and its model lifecycle are the caller's P06/P07 concern.
    The single cleanup callback is invoked exactly once on every exit.
    """
    import torch
    import torch.nn.functional as F
    try:
        if not isinstance(quality, QualityPlan):
            raise ValueError("Expected a validated QualityPlan")
        if type(seed) is not int:
            raise ValueError("Integer global seed required")
        if not callable(predict):
            raise ValueError("Predictor callback required")
        if type(max_cpu_bytes) is not int or type(max_window_bytes) is not int or min(max_cpu_bytes, max_window_bytes) < 1:
            raise ValueError("Positive integer memory budgets required")
        if inference_dtype is None:
            inference_dtype = torch.float32
        # Preflight both stages up front; no failed second-stage surprise after
        # completing expensive first-stage model work.
        _preflight(quality.target, max_cpu_bytes=max_cpu_bytes,
                   max_window_bytes=max_window_bytes)
        if quality.anchor is not None:
            _preflight(quality.anchor, max_cpu_bytes=max_cpu_bytes,
                       max_window_bytes=max_window_bytes)
        if cancelled is not None and cancelled():
            raise InterruptedError("Coherent quality run interrupted")
        stages = []

        def run_stage(name, plan, start, schedule):
            def callback(crop, sigma, tile):
                return predict(crop, sigma, tile, name)
            def report(info):
                if progress is not None:
                    progress(name, info)
            return synchronized_euler(
                plan, start, schedule, callback, device=device,
                inference_dtype=inference_dtype, cancelled=cancelled,
                progress=report, max_cpu_bytes=max_cpu_bytes,
                max_window_bytes=max_window_bytes)

        if quality.mode == "progressive":
            anchor = quality.anchor
            assert anchor is not None and quality.anchor_sigmas is not None
            # Derive two independent deterministic generator seeds without
            # manipulating any global torch RNG state.
            anchor_start = seeded_noise(anchor, seed, sigma=quality.anchor_sigmas[0])
            a = run_stage("anchor", anchor, anchor_start, quality.anchor_sigmas)
            stages.append(QualityStage("anchor", a.plan, a.steps,
                                       a.cpu_workspace_estimate_bytes,
                                       a.max_window_latent_bytes))
            if cancelled is not None and cancelled():
                raise InterruptedError("Coherent quality run interrupted")
            up = F.interpolate(a.latent, size=(quality.target.height, quality.target.width),
                               mode="bilinear", align_corners=False)
            target_noise = seeded_noise(quality.target, seed ^ 0x9E3779B97F4A7C15,
                                        sigma=quality.refine_sigmas[0])
            target_start = up + target_noise
            del up, target_noise, anchor_start, a
            name = "refine"
        else:
            target_start = seeded_noise(quality.target, seed,
                                        sigma=quality.refine_sigmas[0])
            name = "direct"
        result = run_stage(name, quality.target, target_start, quality.refine_sigmas)
        stages.append(QualityStage(name, result.plan, result.steps,
                                   result.cpu_workspace_estimate_bytes,
                                   result.max_window_latent_bytes))
        return QualityResult(result.latent, quality.mode, seed, tuple(stages),
                             sum(len(s.geometry.windows) * len(s.steps) for s in stages))
    finally:
        if cleanup is not None:
            cleanup()
