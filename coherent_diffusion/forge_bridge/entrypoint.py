"""P07 Forge UI generation coordinator: opt-in, never an override of txt2img.

Offline contracts only until real loaded Forge Neo/SDXL/Anima/Q4 testing.
No checkpoint/encoder/extension loader substitutions and no external I/O.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import threading
from dataclasses import dataclass
from typing import Any

from coherent_diffusion.crf import canonical_json, parse_validate
from coherent_diffusion.crf.assets import load_crf_zip

_JOB_LOCK = threading.Lock()
_JOB_CANCEL = threading.Event()
_MAX_B64_CHARS = 34_000_000  # 24 MiB archive with bounded transfer overhead


class UnsupportedGeneration(RuntimeError):
    """Request cannot be executed safely by the currently verified contracts."""


@dataclass(frozen=True)
class JobPlan:
    family: str
    mode: str
    width: int
    height: int
    tile: int
    overlap: int
    halo: int
    seed: int
    steps: int
    cfg: float
    max_cpu_bytes: int
    max_window_bytes: int
    document_hash: str
    has_regions: bool
    quality: Any
    document: dict
    assets: dict


def cancel_generation():
    """Cooperative cancel between steps/windows; no model-global patches."""
    _JOB_CANCEL.set()
    return "Cancellation requested; current model forward may finish first."


def _int(value, label, minimum, maximum):
    if type(value) is bool:
        raise ValueError(f"{label} must be an integer")
    try:
        if isinstance(value, float):
            if not math.isfinite(value) or not value.is_integer():
                raise ValueError()
            n = int(value)
        elif isinstance(value, (int, str)):
            n = int(value)
            if isinstance(value, str) and value.strip() != str(n):
                raise ValueError()
        else:
            raise ValueError()
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if not minimum <= n <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}")
    return n


def _default_doc(prompt, negative, width, height):
    if not isinstance(prompt, str) or not isinstance(negative, str):
        raise ValueError("Scene text must be strings")
    return {
        "format": "coherent-regions", "version": 1,
        "canvas": {"width": width, "height": height,
                   "coordinate_system": "normalized_top_left",
                   "aspect_policy": "preserve_relative"},
        "global": {"scene_prompt": prompt, "style_prompt": "",
                   "negative_prompt": negative, "base_weight": 1.0},
        "blend": {"policy": "weighted"}, "regions": [],
    }


def _parse_composition(text, archive_b64, prompt, negative, width, height):
    if archive_b64 and text:
        raise ValueError("Supply CRF JSON or ZIP, not both")
    if archive_b64:
        if not isinstance(archive_b64, str) or len(archive_b64) > _MAX_B64_CHARS:
            raise ValueError("CRF ZIP transfer exceeds size limit")
        try:
            archive = base64.b64decode(archive_b64, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Invalid CRF ZIP transport encoding") from exc
        imported = load_crf_zip(archive)
        doc, assets = imported.report.document, imported.assets
    elif text:
        if not isinstance(text, str):
            raise ValueError("CRF text must be a string")
        doc = parse_validate(text).document
        assets = {}
        if doc.get("assets"):
            raise ValueError("Binary mask assets require a verified CRF ZIP")
    else:
        doc = _default_doc(prompt, negative, width, height)
        assets = {}
    parse_validate(canonical_json(doc))
    canvas = doc["canvas"]
    if canvas["width"] != width or canvas["height"] != height:
        raise ValueError("CRF canvas must equal output dimensions; implicit resampling is disabled")
    return doc, assets


def _model_sigmas(host, family: str, steps: int):
    """Source-backed per-loaded-predictor timeline; never invent SDXL sigma=1.

    SDXL: sub-sample the installed discrete predictor's training log-sigma
    timeline, including its nonzero terminal training sigma and final zero.
    Anima: use the actual loaded constant-flow time shift, not identity
    timesteps unless the active predictor implements identity.
    """
    import torch
    patcher = getattr(getattr(host, "forge_objects", None), "unet", None)
    predictor = getattr(getattr(patcher, "model", None), "predictor", None)
    if predictor is None or not callable(getattr(predictor, "sigma", None)):
        raise UnsupportedGeneration("Loaded Forge model exposes no verified sigma predictor")
    kind = type(predictor).__name__
    if family == "sdxl":
        if kind != "Prediction" or getattr(predictor, "prediction_type", None) not in (
                "epsilon", "v_prediction"):
            raise UnsupportedGeneration("SDXL predictor is not a supported native discrete Euler source")
        reference = getattr(predictor, "sigmas", None)
        if not isinstance(reference, torch.Tensor) or reference.ndim != 1 or reference.numel() < steps:
            raise UnsupportedGeneration("SDXL predictor's training sigma table is unavailable/too short")
        sample_times = torch.linspace(reference.numel() - 1, 0, steps, dtype=torch.float32)
        values = predictor.sigma(sample_times)
        if not isinstance(values, torch.Tensor) or values.numel() != steps:
            raise UnsupportedGeneration("Unrecognized SDXL predictor sigma result")
        result = tuple(float(v) for v in values.detach().cpu().flatten()) + (0.0,)
    elif family == "anima":
        if kind not in ("PredictionFlux", "PredictionDiscreteFlow") or (
                getattr(predictor, "prediction_type", None) != "const"):
            raise UnsupportedGeneration("Anima requires a verified constant-flow predictor")
        times = torch.linspace(1.0, 0.0, steps + 1)
        if kind == "PredictionDiscreteFlow":
            multiplier = getattr(predictor, "multiplier", None)
            if not isinstance(multiplier, (float, int)) or not math.isfinite(multiplier) or multiplier <= 0:
                raise UnsupportedGeneration("Anima flow time multiplier unavailable")
            times *= multiplier
        values = predictor.sigma(times)
        if not isinstance(values, torch.Tensor) or values.numel() != steps + 1:
            raise UnsupportedGeneration("Unrecognized Anima shifted sigma output")
        result = tuple(float(v) for v in values.detach().cpu().flatten())
        result = result[:-1] + (0.0,)
    else:
        raise UnsupportedGeneration("Unknown model family")
    if (len(result) != steps + 1 or not math.isfinite(result[0])
            or result[0] <= 0 or any(not math.isfinite(x) or x < 0 for x in result)
            or any(a <= b for a, b in zip(result, result[1:]))):
        raise UnsupportedGeneration("Loaded predictor produced invalid/nondecreasing sigma trajectory")
    if family == "anima" and result[0] > 1.0 + 1e-6:
        raise UnsupportedGeneration("Anima native flow sigma exceeds one")
    return result


def preflight(*, host, family, mode, width, height, tile, overlap, halo,
              steps, cfg, seed, cpu_mib, window_mib, scene_json="",
              archive_b64="", prompt="", negative=""):
    """Strict, side-effect-free request preflight before Forge text encoding."""
    from coherent_diffusion.scheduler.sdxl_sync import TilePlan
    from coherent_diffusion.scheduler.quality import QualityPlan
    from coherent_diffusion.scheduler.anima_flow import AnimaQualityPlan
    from coherent_diffusion.forge_bridge.sdxl import ForgeSDXL
    from coherent_diffusion.forge_bridge.anima import ForgeAnima

    if family not in ("sdxl", "anima") or mode not in ("direct", "progressive"):
        raise UnsupportedGeneration("Choose one supported model family and trajectory")
    w = _int(width, "Width", 256, 4096)
    h = _int(height, "Height", 256, 4096)
    if w % 64 or h % 64:
        raise ValueError("Width and height must align to 64 pixels")
    t = _int(tile, "Tile", 256, 1024)
    ov = _int(overlap, "Overlap", 0, 256)
    ha = _int(halo, "Halo", 0, 128)
    if any(x % 64 for x in (t, ov, ha)) or ov >= t:
        raise ValueError("Tile, overlap and halo must be 64px aligned and overlap < tile")
    n = _int(steps, "Steps", 2, 100)
    s = _int(seed, "Seed", 0, 2**63 - 1)
    cm = _int(cpu_mib, "CPU workspace MiB", 128, 65536)
    wm = _int(window_mib, "Window latent MiB", 8, 4096)
    if type(cfg) not in (float, int) and not isinstance(cfg, str):
        raise ValueError("CFG must be a finite nonnegative number")
    try:
        c = float(cfg)
    except (ValueError, OverflowError) as exc:
        raise ValueError("CFG must be numeric") from exc
    if not math.isfinite(c) or not 0 <= c <= 30:
        raise ValueError("CFG must be finite and within [0,30]")
    doc, assets = _parse_composition(scene_json, archive_b64, prompt, negative, w, h)
    if family == "sdxl":
        ForgeSDXL._check_host(host)
    else:
        patch = ForgeAnima.check_host(host)
    if any(r.get("loras") and r.get("enabled", True) for r in doc["regions"]):
        raise UnsupportedGeneration(
            "Regional LoRAs require verified whole-Forge exclusive generation ownership; "
            "P07 real-host acceptance pending. Remove adapters to run other regions.")
    plan = TilePlan.build(w, h, t, ov, ha)
    # Use actual loaded predictor shape and sigma scaling. This is an
    # explicitly selected Euler/flow trajectory, not Forge's default sampler.
    sigmas = _model_sigmas(host, family, n)
    # A separate global stage is necessary for progressive geometry lineage.
    anchor_sigmas = sigmas if mode == "progressive" else None
    if family == "sdxl":
        quality = QualityPlan.build(plan, mode=mode, refine_sigmas=sigmas,
                                    anchor_sigmas=anchor_sigmas)
    else:
        quality = AnimaQualityPlan.build(plan, mode=mode, patch_spatial=patch,
                                         refine_sigmas=sigmas,
                                         anchor_sigmas=anchor_sigmas)
    if assets and not any(r["geometry"]["type"] == "mask" for r in doc["regions"]):
        raise ValueError("Orphan mask data in composition")
    return JobPlan(family, mode, w, h, t, ov, ha, s, n, c,
                   cm * 1024**2, wm * 1024**2,
                   hashlib.sha256(canonical_json(doc).encode("utf-8")).hexdigest(),
                   bool(doc["regions"]), quality, doc, assets)


def _decode(host, plan, latent):
    import torch
    from coherent_diffusion.forge_bridge.sdxl import ForgeSDXL
    from coherent_diffusion.forge_bridge.anima import ForgeAnima

    if plan.family == "sdxl":
        # Decode is a separate host lifecycle phase, after all denoiser hooks
        # have exited. Use the already-loaded native tiled VAE only.
        adapter = ForgeSDXL(host, plan.quality.target, None, None, plan.cfg,
                            plan.seed, "cpu")
    else:
        adapter = ForgeAnima(host, plan.quality.target, None, None, plan.cfg,
                             plan.seed, "cpu", plan.quality.patch_spatial)
    pixels = adapter.decode_tiled(latent)
    if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4:
        raise UnsupportedGeneration("Native Forge VAE returned unsupported image layout")
    if pixels.shape != (1, plan.height, plan.width, 3):
        raise UnsupportedGeneration("Native Forge VAE output dimensions changed")
    if not torch.isfinite(pixels).all():
        raise UnsupportedGeneration("Native Forge VAE produced nonfinite pixels")
    return pixels.float().cpu().clamp(0, 1)


def execute(plan: JobPlan, *, host, cancelled=None, progress=None):
    """Run already-preflighted plan. Returns decoded PIL and portable metadata."""
    if not isinstance(plan, JobPlan):
        raise ValueError("Preflighted JobPlan required")
    if not _JOB_LOCK.acquire(blocking=False):
        raise UnsupportedGeneration("Another Coherent Diffusion job is running")
    _JOB_CANCEL.clear()
    try:
        from coherent_diffusion.forge_bridge.quality import run_forge_sdxl_quality
        from coherent_diffusion.forge_bridge.anima import run_forge_anima_quality
        from coherent_diffusion.forge_bridge.regional import (
            run_forge_regional_sdxl, run_forge_regional_anima)

        def stopped():
            return _JOB_CANCEL.is_set() or bool(cancelled is not None and cancelled())

        if stopped():
            raise InterruptedError("Generation interrupted before sampling")
        keyword = dict(seed=plan.seed, cfg=plan.cfg, cancelled=stopped,
                       progress=progress, max_cpu_bytes=plan.max_cpu_bytes,
                       max_window_bytes=plan.max_window_bytes)
        if plan.has_regions:
            if plan.family == "sdxl":
                result = run_forge_regional_sdxl(
                    host, plan.quality, plan.document, asset_bytes=plan.assets,
                    **keyword)
            else:
                result = run_forge_regional_anima(
                    host, plan.quality, plan.document, asset_bytes=plan.assets,
                    **keyword)
        else:
            gl = plan.document["global"]
            prompt = ", ".join(s.strip() for s in
                               (gl["scene_prompt"], gl["style_prompt"]) if s.strip())
            kwargs = dict(keyword, prompt=prompt,
                          negative_prompt=gl.get("negative_prompt") or "")
            if plan.family == "sdxl":
                result = run_forge_sdxl_quality(host, plan.quality, **kwargs)
            else:
                result = run_forge_anima_quality(host, plan.quality, **kwargs)
        if stopped():
            raise InterruptedError("Generation cancelled before decode")
        pixels = _decode(host, plan, result.latent)
        if stopped():
            raise InterruptedError("Generation cancelled after decode")
        from PIL import Image
        import torch
        frame = (pixels[0] * 255).round().to(torch.uint8).numpy()
        image = Image.fromarray(frame, mode="RGB")
        meta = {
            "format": "coherent-diffusion-replay", "version": 1,
            "verification": "live installation and GPU output require acceptance",
            "family": plan.family, "mode": plan.mode,
            "width": plan.width, "height": plan.height,
            "tile": plan.tile, "overlap": plan.overlap, "halo": plan.halo,
            "steps": plan.steps, "seed": plan.seed, "cfg": plan.cfg,
            "crf_sha256": plan.document_hash,
            "model_forwards_estimate": result.model_forwards,
            "trajectory": "loaded_predictor_uniform_timestep_euler" if plan.family == "sdxl"
                           else "loaded_predictor_shifted_constant_flow",
            "sigma_first": plan.quality.refine_sigmas[0],
            "sigma_sha256": hashlib.sha256(json.dumps(
                plan.quality.refine_sigmas, separators=(",", ":")).encode()).hexdigest(),
            "generation_limits": "batch=1; no hires/img2img/resume; controls/TE-LoRA not activated",
            "workspace_limits_bytes": {"cpu": plan.max_cpu_bytes,
                                       "latent_window": plan.max_window_bytes},
        }
        return image, meta
    finally:
        _JOB_CANCEL.clear()
        _JOB_LOCK.release()


def gradio_generate(scene_json, archive_b64, prompt, negative, family, mode,
                    width, height, tile, overlap, halo, steps, cfg, seed,
                    cpu_mib, window_mib):
    """User-facing callback; return structured error, never a partial image."""
    try:
        from modules import sd_models, shared
        host = sd_models.model_data.sd_model
        if host is None:
            raise UnsupportedGeneration("Load one SDXL or Anima checkpoint in Forge first")
        # This is a cooperative guard, not proof of whole-Forge exclusive ownership.
        state = getattr(shared, "state", None)
        if getattr(state, "job", None):
            raise UnsupportedGeneration("Forge is already processing another job")
        plan = preflight(
            host=host, scene_json=scene_json or "", archive_b64=archive_b64 or "",
            prompt=prompt or "", negative=negative or "", family=family, mode=mode,
            width=width, height=height, tile=tile, overlap=overlap, halo=halo,
            steps=steps, cfg=cfg, seed=seed, cpu_mib=cpu_mib,
            window_mib=window_mib)
        image, details = execute(plan, host=host,
                                 cancelled=lambda: bool(getattr(state, "interrupted", False)))
        return image, details
    except Exception as exc:
        return None, {"status": "error", "type": type(exc).__name__,
                      "detail": str(exc)[:800], "partial_image": False}
