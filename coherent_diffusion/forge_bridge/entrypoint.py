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
    bypassed_region_loras: int = 0
    activate_region_loras: bool = False


def cancel_generation():
    """Cooperative cancel between steps/windows; no model-global patches."""
    from .diagnostics import record as debug_record
    debug_record("ui_cancel_requested")
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


def _validate_requested_loras(doc):
    """Fail before text encoding/sampling if a requested adapter cannot exist.

    Use only Forge's locally registered aliases; no external downloads, text
    encoder patching or guessed filesystem paths. Native loader still verifies
    hashes and patcher success when the actual regional state is entered.
    """
    import importlib
    from pathlib import Path
    from coherent_diffusion.forge_bridge.regional import UnsupportedRegional
    try:
        registry = importlib.import_module("networks")
    except ImportError as exc:
        raise UnsupportedRegional(
            "Forge's built-in LoRA registry is not available; cannot activate regional LoRAs") from exc
    aliases = getattr(registry, "available_network_aliases", None)
    if not isinstance(aliases, dict):
        raise UnsupportedRegional("Forge's available_network_aliases registry is not ready")
    for region in doc["regions"]:
        if not region.get("enabled", True) or region["blend"]["weight"] <= 0:
            continue
        for adapter in region.get("loras", []):
            if adapter["requested_isolation"] == "experimental_internal":
                raise UnsupportedRegional(
                    f"Region {region['id']}: experimental internal LoRA isolation is unavailable")
            if adapter.get("te_weight") not in (None, 0):
                raise UnsupportedRegional(
                    f"Region {region['id']}: text-encoder LoRA cannot mutate a shared/Q4 encoder")
            name = adapter["name"]
            entry = aliases.get(name)
            filename = getattr(entry, "filename", None)
            if not isinstance(filename, str) or not Path(filename).is_file():
                raise UnsupportedRegional(
                    f"Region {region['id']}: LoRA '{name}' is not in Forge's local registry "
                    "or its file is unavailable. Choose a registered local LoRA.")


def preflight(*, host, family, mode, width, height, tile, overlap, halo,
              steps, cfg, seed, cpu_mib, window_mib, scene_json="",
              archive_b64="", prompt="", negative="", activate_region_loras=False):
    """Strict, side-effect-free request preflight before Forge text encoding."""
    from coherent_diffusion.scheduler.sdxl_sync import TilePlan
    from coherent_diffusion.scheduler.quality import QualityPlan
    from coherent_diffusion.scheduler.anima_flow import AnimaQualityPlan
    from coherent_diffusion.forge_bridge.sdxl import ForgeSDXL
    from coherent_diffusion.forge_bridge.anima import ForgeAnima

    if family not in ("sdxl", "anima") or mode not in ("direct", "progressive"):
        raise UnsupportedGeneration("Choose one supported model family and trajectory")
    w = _int(width, "Width", 256, 16384)
    h = _int(height, "Height", 256, 16384)
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
    if type(activate_region_loras) is not bool:
        raise ValueError("Regional LoRA activation flag must be boolean")
    # Hash the *authored* CRF before compiling a transient execution copy.
    # The editor's starter historically referenced a non-existent example LoRA.
    # Such authoring intent must never silently trigger native model mutations.
    authored_hash = hashlib.sha256(canonical_json(doc).encode("utf-8")).hexdigest()
    active_refs = sum(len(r.get("loras", [])) for r in doc["regions"]
                      if r.get("enabled", True) and r["blend"]["weight"] > 0)
    bypassed = active_refs if not activate_region_loras else 0
    if bypassed:
        # parse_validate already returned a newly decoded doc. Do not change
        # source archives/drafts; mode and count are explicit in UI/evidence.
        doc = {**doc, "regions": [
            {**r, "loras": []} if r.get("loras") else r
            for r in doc["regions"]
        ]}
    elif active_refs:
        _validate_requested_loras(doc)

    plan = TilePlan.build(w, h, t, ov, ha)
    # Use actual loaded predictor shape and sigma scaling. This is an
    # explicitly selected Euler/flow trajectory, not Forge's default sampler.
    sigmas = _model_sigmas(host, family, n)
    # A progressive second stage MUST start at an intermediate model-native
    # sigma. Starting Anima refinement at sigma=1 erases the upscaled global
    # anchor by the native flow mixing equation: (1-sigma)*anchor + sigma*noise.
    # SDXL also benefits from a partial-denoise refinement instead of
    # overwhelming its anchor with full-training-sigma noise again.
    if mode == "progressive":
        cut = max(1, (len(sigmas) - 1) // 2)
        anchor_sigmas = sigmas
        refine_sigmas = sigmas[cut:]
    else:
        anchor_sigmas = None
        refine_sigmas = sigmas
    if family == "sdxl":
        quality = QualityPlan.build(plan, mode=mode, refine_sigmas=refine_sigmas,
                                    anchor_sigmas=anchor_sigmas)
    else:
        quality = AnimaQualityPlan.build(plan, mode=mode, patch_spatial=patch,
                                         refine_sigmas=refine_sigmas,
                                         anchor_sigmas=anchor_sigmas)
    if assets and not any(r["geometry"]["type"] == "mask" for r in doc["regions"]):
        raise ValueError("Orphan mask data in composition")
    return JobPlan(family, mode, w, h, t, ov, ha, s, n, c,
                   cm * 1024**2, wm * 1024**2,
                   authored_hash,
                   bool(doc["regions"]), quality, doc, assets, bypassed,
                   activate_region_loras)


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
    # Native VAE returns a full NHWC image. Avoid an extra full-size FP32
    # copy; assemble display-ready bytes in bounded row stripes below.
    return pixels if pixels.device.type == "cpu" else pixels.to("cpu")


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
        import numpy as np
        frame = np.empty((plan.height, plan.width, 3), dtype=np.uint8)
        # Native tiled VAE may still hold multiple complete FP32 CPU tensors.
        # Row-stripe conversion avoids additional full-frame float temporaries.
        for top in range(0, plan.height, 64):
            if stopped():
                raise InterruptedError("Generation cancelled while assembling decoded image")
            bottom = min(plan.height, top + 64)
            stripe = pixels[0, top:bottom].float().clamp(0, 1)
            frame[top:bottom] = (stripe * 255).round().to(torch.uint8).numpy()
        del pixels
        image = Image.fromarray(frame)
        meta = {
            "format": "coherent-diffusion-replay", "version": 1,
            "verification": "live installation and GPU output require acceptance",
            "family": plan.family, "mode": plan.mode,
            "width": plan.width, "height": plan.height,
            "tile": plan.tile, "overlap": plan.overlap, "halo": plan.halo,
            "global_anchor": ({"width": plan.quality.anchor.width_px,
                               "height": plan.quality.anchor.height_px,
                               "effective_factor": (plan.quality.anchor_factor if plan.family == "sdxl"
                                                    else plan.quality.factor),
                               "actual_model_forwards": len(plan.quality.anchor.windows)}
                              if plan.quality.anchor is not None else None),
            "steps": plan.steps, "seed": plan.seed, "cfg": plan.cfg,
            "crf_sha256": plan.document_hash,
            "active_regional_loras": plan.activate_region_loras,
            "regional_lora_references_bypassed": plan.bypassed_region_loras,
            "window_step_evaluations_lower_bound": result.model_forwards,
            "trajectory": "loaded_predictor_uniform_timestep_euler" if plan.family == "sdxl"
                           else "loaded_predictor_shifted_constant_flow",
            "sigma_first": plan.quality.refine_sigmas[0],
            "refine_steps": len(plan.quality.refine_sigmas) - 1,
            "anchor_steps": (len(plan.quality.anchor_sigmas) - 1
                             if plan.quality.anchor_sigmas is not None else 0),
            "anchor_sigma_first": (plan.quality.anchor_sigmas[0]
                                   if plan.quality.anchor_sigmas is not None else None),
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
                    cpu_mib, window_mib, activate_region_loras=False):
    """Native Forge UI job: queued with Forge txt2img and its checkpoint loader.

    We do not replace the installed Anima/Q4 encoder or patcher lifecycle.
    The per-run model-object pointer is restored on every exit path. Actual
    GPU/encoder correctness still requires installed-host P07 acceptance.
    """
    from .diagnostics import record as debug_record
    stage = "startup"
    started = __import__("time").monotonic()
    debug_record("generation_started", family=family, mode=mode,
                 width=width, height=height, steps=steps, tile=tile)
    try:
        from modules import sd_models, shared
        if not hasattr(sd_models, "model_data"):
            raise UnsupportedGeneration("Forge model loader unavailable")
        # Offline no-host probes and actual missing-checkpoint are distinct
        # from the Forge FakeInitialModel placeholder (loadable by UI).
        if getattr(sd_models.model_data, "sd_model", None) is None:
            raise UnsupportedGeneration("Load an SDXL or Anima checkpoint by selecting it in Forge first")
        from modules import call_queue
        from modules_forge import main_entry

        # All native txt2img and img2img jobs use this same FIFO lock. Gradio's
        # own queue does not exclude concurrent tabs or extensions by itself.
        with call_queue.queue_lock:
            stage = "native_job_begin"
            debug_record("generation_stage", phase=stage)
            state = getattr(shared, "state", None)
            if state is None:
                raise UnsupportedGeneration("Forge job state unavailable")
            state.begin(job="coherent-diffusion")
            try:
                # Resolve *current* native checkpoint and additional modules;
                # trigger the same native reload as ordinary Forge generation.
                stage = "native_model_load"
                debug_record("generation_stage", phase=stage)
                main_entry.refresh_model_loading_parameters()
                host, _ = sd_models.forge_model_reload()
                if host is None or type(host).__name__ == "FakeInitialModel":
                    raise UnsupportedGeneration(
                        "Forge has no loaded checkpoint; select a model and inspect native load logs")
                original_holder = getattr(host, "forge_objects", None)
                pristine_holder = getattr(host, "forge_objects_original", None)
                if original_holder is None or pristine_holder is None or not callable(
                        getattr(pristine_holder, "shallow_copy", None)):
                    raise UnsupportedGeneration("Loaded Forge original model patchers unavailable")
                # A native generation may leave globally adapted patcher
                # pointers active. Use a scoped shallow copy of the original
                # patcher set for our own region states, restore it afterwards.
                host.forge_objects = pristine_holder.shallow_copy()
                try:
                    stage = "preflight"
                    debug_record("generation_stage", phase=stage)
                    plan = preflight(
                        host=host, scene_json=scene_json or "",
                        archive_b64=archive_b64 or "", prompt=prompt or "",
                        negative=negative or "", family=family, mode=mode,
                        width=width, height=height, tile=tile, overlap=overlap,
                        halo=halo, steps=steps, cfg=cfg, seed=seed,
                        cpu_mib=cpu_mib, window_mib=window_mib,
                        activate_region_loras=activate_region_loras)
                    stage = "inference_and_decode"
                    debug_record("generation_stage", phase=stage)
                    image, details = execute(
                        plan, host=host,
                        cancelled=lambda: bool(getattr(state, "interrupted", False)))
                    debug_record("generation_completed", success=True,
                                 elapsed_ms=(__import__("time").monotonic()-started)*1000,
                                 width=getattr(image, "width", None),
                                 height=getattr(image, "height", None))
                    return image, details
                finally:
                    host.forge_objects = original_holder
            finally:
                state.end()
    except Exception as exc:
        debug_record("generation_error", type=type(exc).__name__, phase=stage,
                     elapsed_ms=(__import__("time").monotonic()-started)*1000)
        # The Forge terminal gets a traceback for actual integration defects.
        # Do not include prompts, model weights or local input paths in JSON.
        import traceback
        print(f"[Coherent Diffusion] {stage}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None, {"status": "error", "type": type(exc).__name__,
                      "phase": stage,
                      "detail": str(exc)[:800], "partial_image": False}
