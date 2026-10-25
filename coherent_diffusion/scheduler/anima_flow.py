"""P05 Anima one-frame WAN latent coherent flow sampler (offline contract).

Different from P03 SDXL: [B,16,T=1,H,W] and constant-flow x0 in [0,1].
The host returns Forge KModel CFG-combined denoised x0, NOT raw velocity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from .sdxl_sync import Tile, TilePlan, _weights
from .anchor import anchor_dimensions, expected_anchor_dimensions


def flow_sigmas(values) -> tuple[float, ...]:
    try:
        v = tuple(float(s) for s in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Anima requires explicit numeric flow sigmas") from exc
    if len(v) < 2 or v[-1] != 0.0 or not (0.0 < v[0] <= 1.0):
        raise ValueError("Anima flow sigmas must start within (0,1] and end at zero")
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s in v):
        raise ValueError("Nonfinite/out-of-range flow sigma")
    if any(a <= b for a, b in zip(v, v[1:])):
        raise ValueError("Anima flow sigmas must strictly decrease")
    return v


def validate_anima_tiles(plan: TilePlan, *, patch_spatial: int) -> None:
    if not isinstance(plan, TilePlan) or plan.scale != 8:
        raise ValueError("Anima requires eight-pixel WAN spatial latent scale")
    if type(patch_spatial) is not int or patch_spatial <= 0 or patch_spatial > 16:
        raise ValueError("Unsupported Anima DiT spatial patch size")
    expected = TilePlan.build(plan.width_px, plan.height_px, plan.tile_px,
                              plan.overlap_px, plan.halo_px, scale=8)
    if len(expected.windows) != len(plan.windows) or set(expected.windows) != set(plan.windows):
        raise ValueError("Incomplete/altered Anima windows")
    for t in plan.windows:
        r = t.read
        if (r.x0 % patch_spatial or r.y0 % patch_spatial
                or r.width % patch_spatial or r.height % patch_spatial):
            raise ValueError("Anima read windows must align to absolute DiT patch coordinates")


def anima_noise(plan: TilePlan, seed: int, sigma: float, *, batch: int = 1):
    if type(seed) is not int or not 0 <= seed < 2**63 or type(batch) is not int or batch < 1:
        raise ValueError("Anima needs nonnegative signed seed and positive batch")
    if not math.isfinite(sigma) or not 0 < sigma <= 1:
        raise ValueError("Flow initial sigma out of range")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((batch, 16, 1, plan.height, plan.width),
                       generator=rng, dtype=torch.float32, device="cpu") * sigma


def _budget(plan: TilePlan, batch: int, max_cpu_bytes: int, max_window_bytes: int):
    if (type(max_cpu_bytes) is not int or type(max_window_bytes) is not int
            or min(max_cpu_bytes, max_window_bytes) <= 0):
        raise ValueError("Positive integral CPU/window budgets required")
    field = batch * 16 * plan.height * plan.width * 4
    cpu = 4 * field + plan.height * plan.width * 4
    maximum = max(batch * 16 * t.read.height * t.read.width * 4 for t in plan.windows)
    if cpu > max_cpu_bytes or maximum > max_window_bytes:
        raise MemoryError("Anima CPU/global or individual window memory budget exceeded")
    return cpu, maximum


@dataclass(frozen=True)
class AnimaStep:
    index: int
    sigma: float
    next_sigma: float
    windows: int


@dataclass(frozen=True)
class AnimaResult:
    latent: torch.Tensor
    steps: tuple[AnimaStep, ...]
    plan: TilePlan
    cpu_workspace_estimate_bytes: int
    max_window_latent_bytes: int


def synchronized_flow(plan: TilePlan, latent: torch.Tensor, sigmas,
                      predict: Callable[[torch.Tensor, float, Tile], torch.Tensor],
                      *, patch_spatial: int, device="cpu",
                      inference_dtype=torch.float32, cancelled=None, progress=None,
                      cleanup=None, max_cpu_bytes=4 * 1024**3,
                      max_window_bytes=512 * 1024**2) -> AnimaResult:
    """One CPU-resident 5D global latent, immutable window snapshot per flow step."""
    try:
        validate_anima_tiles(plan, patch_spatial=patch_spatial)
        schedule = flow_sigmas(sigmas)
        if (not isinstance(latent, torch.Tensor) or latent.device.type != "cpu"
                or latent.dtype != torch.float32 or latent.ndim != 5
                or tuple(latent.shape[1:]) != (16, 1, plan.height, plan.width)
                or latent.shape[0] < 1 or not bool(torch.isfinite(latent).all())):
            raise ValueError("Anima needs finite CPU float32 latent [B,16,1,H,W]")
        if inference_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Unsupported Anima compute dtype")
        if not callable(predict):
            raise ValueError("Anima x0 predictor required")
        cpu, window = _budget(plan, latent.shape[0], max_cpu_bytes, max_window_bytes)
        ordered = sorted(plan.windows, key=lambda t: (t.write.y0, t.write.x0, t.index))
        state = latent.clone()
        steps = []
        for i, (s, ns) in enumerate(zip(schedule, schedule[1:])):
            if cancelled is not None and cancelled():
                raise InterruptedError("Anima flow cancelled")
            frozen = state
            numerator = torch.zeros_like(frozen)
            denominator = torch.zeros((1, 1, 1, plan.height, plan.width), dtype=torch.float32)
            for tile in ordered:
                if cancelled is not None and cancelled():
                    raise InterruptedError("Anima flow cancelled")
                r, q = tile.read, tile.write
                crop = frozen[..., r.y0:r.y1, r.x0:r.x1].clone().to(
                    device=device, dtype=inference_dtype)
                try:
                    with torch.inference_mode():
                        prediction = predict(crop, s, tile)
                    if not isinstance(prediction, torch.Tensor) or prediction.shape != crop.shape:
                        raise ValueError("Anima predicted x0 shape mismatch")
                    out = prediction.detach().to(device="cpu", dtype=torch.float32, copy=True)
                    if not bool(torch.isfinite(out).all()):
                        raise ValueError("Nonfinite Anima x0")
                finally:
                    del crop
                oy, ox = q.y0 - r.y0, q.x0 - r.x0
                weight = _weights(tile, plan).unsqueeze(2)
                numerator[..., q.y0:q.y1, q.x0:q.x1] += (
                    out[..., oy:oy + q.height, ox:ox + q.width] * weight)
                denominator[..., q.y0:q.y1, q.x0:q.x1] += weight
                del out
            if not bool(torch.all(denominator > 0)):
                raise ValueError("Anima tile coverage gap")
            x0 = numerator / denominator
            # KModel constant-flow predictor returns x0=x-sigma*velocity.
            state = frozen + (frozen - x0) * ((ns - s) / s)
            if not bool(torch.isfinite(state).all()):
                raise ValueError("Anima flow produced nonfinite global field")
            info = AnimaStep(i, s, ns, len(ordered))
            steps.append(info)
            if progress is not None:
                progress(info)
        return AnimaResult(state, tuple(steps), plan, cpu, window)
    finally:
        if cleanup is not None:
            cleanup()


@dataclass(frozen=True)
class AnimaQualityPlan:
    mode: str
    target: TilePlan
    refine_sigmas: tuple[float, ...]
    anchor: TilePlan | None
    anchor_sigmas: tuple[float, ...] | None
    patch_spatial: int
    factor: int

    @classmethod
    def build(cls, target: TilePlan, *, patch_spatial: int, mode: str,
              refine_sigmas, anchor_sigmas=None, factor: int = 2,
              max_anchor_pixels: int | None = None, global_interaction: str = "none"):
        validate_anima_tiles(target, patch_spatial=patch_spatial)
        if global_interaction != "none":
            raise ValueError("Unvalidated Anima global interactions are disabled")
        if mode not in ("direct", "progressive") or type(factor) is not int or factor not in (2, 4):
            raise ValueError("Unsupported Anima quality mode/anchor factor")
        refine = flow_sigmas(refine_sigmas)
        if mode == "direct":
            if anchor_sigmas is not None:
                raise ValueError("Direct mode cannot run hidden anchor")
            return cls(mode, target, refine, None, None, patch_spatial, factor)
        if anchor_sigmas is None:
            raise ValueError("Progressive requires explicit global overview sigmas")
        w, h, effective_factor = anchor_dimensions(
            target, factor, alignment_px=target.scale * patch_spatial,
            max_anchor_pixels=max_anchor_pixels)
        anchor = TilePlan.build(w, h, max(w, h), 0, 0, scale=8)
        validate_anima_tiles(anchor, patch_spatial=patch_spatial)
        if len(anchor.windows) != 1:
            raise AssertionError("Anchor must be one global DiT window")
        return cls(mode, target, refine, anchor, flow_sigmas(anchor_sigmas),
                   patch_spatial, effective_factor)


@dataclass(frozen=True)
class AnimaQualityResult:
    latent: torch.Tensor
    mode: str
    stages: tuple[tuple[str, AnimaResult], ...]
    model_forwards: int


def execute_anima(plan: AnimaQualityPlan, seed: int, predict, *, device="cpu",
                  inference_dtype=torch.float32, cancelled=None, progress=None,
                  cleanup=None, max_cpu_bytes=4 * 1024**3,
                  max_window_bytes=512 * 1024**2):
    """Distinct flow-native quality runner; compatible mode policy with P04."""
    try:
        if not isinstance(plan, AnimaQualityPlan) or not callable(predict):
            raise ValueError("Anima validated quality plan/predictor required")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Anima requires signed 63-bit seed")
        validate_anima_tiles(plan.target, patch_spatial=plan.patch_spatial)
        flow_sigmas(plan.refine_sigmas)
        _budget(plan.target, 1, max_cpu_bytes, max_window_bytes)
        if plan.mode == "progressive":
            if plan.anchor is None or plan.anchor_sigmas is None:
                raise ValueError("Missing Anima anchor")
            validate_anima_tiles(plan.anchor, patch_spatial=plan.patch_spatial)
            if ((plan.anchor.width_px, plan.anchor.height_px) !=
                    expected_anchor_dimensions(plan.target, plan.factor,
                                               plan.target.scale * plan.patch_spatial)
                    or len(plan.anchor.windows) != 1):
                raise ValueError("Anima progressive anchor geometry/lineage mismatch")
            flow_sigmas(plan.anchor_sigmas)
            _budget(plan.anchor, 1, max_cpu_bytes, max_window_bytes)
            target_field = 16 * plan.target.height * plan.target.width * 4
            anchor_field = 16 * plan.anchor.height * plan.anchor.width * 4
            if 7 * target_field + anchor_field + plan.target.height * plan.target.width * 4 > max_cpu_bytes:
                raise MemoryError("Anima progressive CPU handoff budget exceeded")
        elif plan.mode != "direct" or plan.anchor is not None or plan.anchor_sigmas is not None:
            raise ValueError("Unvalidated Anima quality plan")
        if cancelled is not None and cancelled():
            raise InterruptedError("Anima flow cancelled")
        stages = []

        def stage(name, geometry, start, schedule):
            def sample(crop, sigma, tile):
                return predict(crop, sigma, tile, name)
            def report(s):
                if progress is not None:
                    progress(name, s)
            result = synchronized_flow(
                geometry, start, schedule, sample, patch_spatial=plan.patch_spatial,
                device=device, inference_dtype=inference_dtype, cancelled=cancelled,
                progress=report, max_cpu_bytes=max_cpu_bytes,
                max_window_bytes=max_window_bytes)
            stages.append((name, result))
            return result.latent

        if plan.mode == "progressive":
            anchor = plan.anchor
            assert anchor is not None and plan.anchor_sigmas is not None
            anchor_start = anima_noise(anchor, seed, plan.anchor_sigmas[0])
            a = stage("anchor", anchor, anchor_start, plan.anchor_sigmas)
            if cancelled is not None and cancelled():
                raise InterruptedError("Anima flow cancelled")
            lifted = F.interpolate(a.squeeze(2), size=(plan.target.height, plan.target.width),
                                   mode="bilinear", align_corners=False).unsqueeze(2)
            noise = anima_noise(plan.target, seed ^ 0x1E3779B97F4A7C15, plan.refine_sigmas[0])
            # Native KModel constant-flow noise scaling: (1-sigma)*x0 + sigma*epsilon.
            # Unlike SDXL Euler, flow must NOT simply add noise to the anchor.
            start = (1.0 - plan.refine_sigmas[0]) * lifted + noise
            del anchor_start, a, lifted, noise
            final_stage = "refine"
        else:
            start = anima_noise(plan.target, seed, plan.refine_sigmas[0])
            final_stage = "direct"
        result = stage(final_stage, plan.target, start, plan.refine_sigmas)
        return AnimaQualityResult(result, plan.mode, tuple(stages),
                                  sum(len(r.plan.windows) * len(r.steps) for _, r in stages))
    finally:
        if cleanup is not None:
            cleanup()
