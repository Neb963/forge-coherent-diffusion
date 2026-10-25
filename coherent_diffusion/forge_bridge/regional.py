"""P06 opt-in source-backed Forge regional prediction bridge.

Never installs global hooks or changes Forge's Q4 text loader. Distinct prompt/
denoiser LoRA states are serial, mask-blended against the SAME crop and sigma.
The installation's real model, LoRA format and GPU remain P07 evidence.
"""
from __future__ import annotations

import hashlib
import importlib
import threading
from contextlib import contextmanager
from pathlib import Path

from coherent_diffusion.forge_bridge.anima import ForgeAnima
from coherent_diffusion.forge_bridge.sdxl import ForgeSDXL
from coherent_diffusion.scheduler.anima_flow import AnimaQualityPlan, execute_anima
from coherent_diffusion.scheduler.guidance import GuidancePlan
from coherent_diffusion.scheduler.quality import QualityPlan, execute_quality
from coherent_diffusion.scheduler.regional import (
    PredictionState, UnsupportedRegional, blended_prediction, compile_regional,
)

_STATE_LOCK = threading.Lock()


def _join(*parts):
    return ", ".join(x.strip() for x in parts if isinstance(x, str) and x.strip())


class NativeForgeLoRAs:
    """Resolve symbolic names ONLY through the installed Forge LoRA registry.

    Native loader returns cloned model patchers, not filenames to be imported
    from CRF. No TE patcher replacement; Q4/Qwen path remains Forge-owned.
    The host must be exclusively owned; other running Forge jobs are not
    serialized by this extension lock (P07 integration gate).
    """
    def __init__(self, registry_module="networks"):
        self.registry_module = registry_module

    @contextmanager
    def scoped(self, host, adapters):
        if not adapters:
            yield
            return
        if not _STATE_LOCK.acquire(blocking=False):
            raise UnsupportedRegional("Another regional LoRA evaluation is active")
        obj = getattr(host, "forge_objects", None)
        if obj is None or not hasattr(obj, "unet") or not hasattr(obj, "clip"):
            _STATE_LOCK.release()
            raise UnsupportedRegional("Host patchers unavailable for LoRA transaction")
        original_unet, original_clip = obj.unet, obj.clip
        try:
            module = importlib.import_module(self.registry_module)
            from backend.args import dynamic_args
            if getattr(dynamic_args, "nunchaku", False):
                raise UnsupportedRegional("Nunchaku shared LoRA mutation is unsupported")
            aliases = getattr(module, "available_network_aliases", None)
            if not isinstance(aliases, dict):
                raise UnsupportedRegional("Forge LoRA registry unavailable")
            source_objects = getattr(host, "forge_objects_original", None)
            if source_objects is not None and getattr(source_objects, "unet", original_unet) is not original_unet:
                raise UnsupportedRegional("Existing globally patched LoRAs prevent regional isolation")
            unet = original_unet
            names = set()
            for name, expected_sha, weight in adapters:
                if name in names:
                    raise UnsupportedRegional("Duplicate LoRA name within one model state")
                names.add(name)
                entry = aliases.get(name)
                if entry is None or not isinstance(getattr(entry, "filename", None), str):
                    raise UnsupportedRegional("LoRA symbolic reference not present in local Forge registry: " + name)
                path = Path(entry.filename)
                if not path.is_file():
                    raise UnsupportedRegional("Registered LoRA file is unavailable: " + name)
                if expected_sha:
                    digest = hashlib.sha256()
                    with path.open("rb") as fd:
                        for block in iter(lambda: fd.read(1024 * 1024), b""):
                            digest.update(block)
                    if digest.hexdigest() != expected_sha:
                        raise UnsupportedRegional("Registered LoRA SHA256 mismatch: " + name)
                state_dict = module.load_lora_state_dict(str(path))
                # Do not pass a CLIP patcher: regional TE-LoRAs would modify
                # the encoder used by all other spatial states.
                unet_new, clip_new = module.load_lora_for_models(
                    unet, None, state_dict, weight, 0.0, filename=str(path), online_mode=False)
                if unet_new is unet or clip_new is not None:
                    raise UnsupportedRegional("Native Forge LoRA did not yield a distinct denoiser patcher")
                unet = unet_new
            if obj.unet is not original_unet or obj.clip is not original_clip:
                raise UnsupportedRegional("Forge host patchers changed during adapter resolution")
            obj.unet = unet
            try:
                yield
            finally:
                # Always restore original loaded patcher/encoder even on OOM.
                obj.unet = original_unet
                obj.clip = original_clip
        finally:
            _STATE_LOCK.release()


class _RegionalRunner:
    def __init__(self, host, quality, doc, *, model, cfg, seed,
                 device, asset_bytes, loras, cancelled, max_forwards_per_window,
                 guidance=(), control_factory=None, text_mode="prediction"):
        import torch
        self.host = host
        self.quality = quality
        self.model = model
        self.cfg = cfg
        self.seed = seed
        self.cancelled = cancelled
        self.device = device
        self.loras = loras
        self.cached = {}
        self.active = None
        self.active_scope = None
        self.active_key = None
        self.regional = compile_regional(
            doc, quality.target, asset_bytes=asset_bytes,
            max_forwards_per_window=max_forwards_per_window,
            include_scene_in_base=quality.mode == "direct")
        if text_mode not in ("prediction", "native_sdxl_masks"):
            raise UnsupportedRegional("Unknown regional text execution mode")
        if text_mode == "native_sdxl_masks" and (model != "sdxl" or guidance):
            raise UnsupportedRegional("Native condition-mask batching is SDXL-only without structural controls")
        if text_mode == "native_sdxl_masks":
            names = [w.state for w in self.regional.states]
            if any(x.adapters for x in names) or len({x.negative for x in names}) != 1:
                raise UnsupportedRegional(
                    "Native SDXL mask batching needs identical negative and no scoped LoRAs; use prediction mode")
        self.text_mode = text_mode
        self.guidance = GuidancePlan(doc, quality.target, model, tuple(guidance),
                                     asset_bytes=asset_bytes)
        if self.guidance.requests and quality.mode == "progressive":
            raise UnsupportedRegional(
                "Progressive anchor guidance requires a separately validated anchor map; "
                "use direct mode or omit structural guidance")
        if self.guidance.requests and not callable(control_factory):
            raise UnsupportedRegional(
                "Structural guidance requires explicit supported native Forge control factory")
        self.control_factory = control_factory
        # Anima Q4/SDXL global encoder is never replaced. Global objects
        # are placed only in progressive anchor; tiles receive shared style.
        self.anchor = PredictionState(_join(doc["global"]["scene_prompt"],
                                            doc["global"]["style_prompt"]),
                                      doc["global"].get("negative_prompt") or "")

    def close(self):
        try:
            if self.active is not None:
                self.active.cleanup()
        finally:
            self.active = None
            self.active_key = None
            if self.active_scope is not None:
                scope, self.active_scope = self.active_scope, None
                scope.__exit__(None, None, None)

    def predict_state(self, state, crop, sigma, tile, stage):
        if self.cancelled is not None and self.cancelled():
            raise InterruptedError("Regional Forge generation cancelled")
        key = (stage, state)
        if key != self.active_key:
            self.close()
            if key not in self.cached:
                selected = self.quality.anchor if stage == "anchor" else self.quality.target
                if selected is None:
                    raise UnsupportedRegional("Missing global anchor geometry")
                cls = ForgeSDXL if self.model == "sdxl" else ForgeAnima
                kwargs = dict(prompt=state.prompt, negative_prompt=state.negative,
                              cfg=self.cfg, seed=self.seed, device=self.device)
                # Compile through existing Forge encoder; no regional TE-LoRAs.
                self.cached[key] = cls.from_prompts(self.host, selected, **kwargs)
            adapter = self.cached[key]
            if state.adapters:
                scope = self.loras.scoped(self.host, state.adapters)
                scope.__enter__()
                self.active_scope = scope
            try:
                adapter.prepare()
                self.active = adapter
                self.active_key = key
            except BaseException:
                self.close()
                raise
        if stage == "anchor" or not self.guidance.requests:
            return self.active.predict(crop, sigma, tile)
        start_sigma = self.quality.refine_sigmas[0]
        fraction = max(0.0, min(1.0, 1.0 - float(sigma)/float(start_sigma)))
        windows = self.guidance.for_window(tile, fraction)
        if not windows:
            return self.active.predict(crop, sigma, tile)
        # Factory owns native ControlNet/LLLite/Anima control activation, VRAM
        # reservation and cleanup; do not load another control stack.
        with self.control_factory(self.host, windows, state, tile) as native:
            if not callable(getattr(native, "get_control", None)):
                raise UnsupportedRegional("Factory did not supply a native Forge control")
            original_positive, original_negative = self.active.positive, self.active.negative
            if not isinstance(original_positive, (list, tuple)) or not isinstance(original_negative, (list, tuple)):
                raise UnsupportedRegional("Unsupported Forge conditions for control attachment")
            if not all(isinstance(x, dict) and "control" not in x for x in
                       list(original_positive) + list(original_negative)):
                raise UnsupportedRegional("Conflicting/unsupported native condition hooks")
            self.active.positive = [dict(x, control=native) for x in original_positive]
            self.active.negative = [dict(x, control=native) for x in original_negative]
            try:
                return self.active.predict(crop, sigma, tile)
            finally:
                self.active.positive = original_positive
                self.active.negative = original_negative

    def predict_native_masked(self, crop, sigma, tile):
        """Forge's native masked conditional batch, 2D SDXL only.

        Native calc_cond_uncond_batch may still execute multiple denoiser calls
        under tight VRAM; this is NOT a guaranteed single model forward nor
        spatial isolation of individual internal attention activations.
        """
        import torch
        from backend.sampling.sampling_function import sampling_function_inner
        entries = self.regional.window(tile)
        if not entries:
            raise UnsupportedRegional("Native mask batch has no spatial coverage")
        conditions = []
        total = len(entries)
        for state, mask in entries:
            key = ("direct" if self.quality.mode == "direct" else "refine", state)
            if key not in self.cached:
                self.cached[key] = ForgeSDXL.from_prompts(
                    self.host, self.quality.target, prompt=state.prompt,
                    negative_prompt=state.negative, cfg=self.cfg, seed=self.seed,
                    device=self.device)
            adapter = self.cached[key]
            if (not isinstance(adapter.positive, (list, tuple)) or len(adapter.positive) != 1
                    or not isinstance(adapter.negative, (list, tuple))
                    or len(adapter.negative) != 1):
                raise UnsupportedRegional("Native SDXL multi-cond path needs one compiled entry per prompt")
            entry = adapter.positive[0]
            if (not isinstance(entry, dict) or any(x in entry for x in
                    ("mask", "strength", "control", "patches"))):
                raise UnsupportedRegional("Conflicting Forge condition patch/mask hooks")
            # Forge sums entry.strength for its CFG edit strength. Equal
            # strength 1/N keeps CFG unchanged; pixel-normalized masks retain
            # the desired *relative* P06 shares after division by out_count.
            conditions.append(dict(entry, mask=mask.to(
                device=crop.device, dtype=crop.dtype).unsqueeze(0),
                strength=1.0/total))
        name = ("native_sdxl_masks", self.quality.mode)
        if self.active_key != name:
            self.close()
            adapter = self.cached[("direct" if self.quality.mode == "direct" else "refine",
                                   entries[0][0])]
            adapter.prepare()
            self.active = adapter
            self.active_key = name
        noise = torch.full((crop.shape[0],), float(sigma), device=crop.device,
                           dtype=crop.dtype)
        return sampling_function_inner(
            self.host.forge_objects.unet.model, crop, noise,
            self.active.negative, conditions, self.cfg, {}, self.seed)

    def predict(self, crop, sigma, tile, stage):
        if stage == "anchor":
            return self.predict_state(self.anchor, crop, sigma, tile, stage)
        if self.text_mode == "native_sdxl_masks":
            return self.predict_native_masked(crop, sigma, tile)
        return blended_prediction(
            self.regional, tile, crop, sigma, stage,
            self.predict_state, cancelled=self.cancelled)


def run_forge_regional_sdxl(host, quality: QualityPlan, doc: dict, *, cfg: float,
                            seed: int, asset_bytes=None, loras=None, device=None,
                            inference_dtype=None, cancelled=None, progress=None,
                            max_cpu_bytes=4 * 1024**3,
                            max_window_bytes=512 * 1024**2,
                            max_forwards_per_window=16, guidance=(), control_factory=None,
                            text_mode="prediction"):
    """Return P04 QualityResult; decoder remains ForgeSDXL.decode_tiled."""
    if not isinstance(quality, QualityPlan):
        raise ValueError("SDXL QualityPlan required")
    if device is None:
        from backend import memory_management
        device = memory_management.get_torch_device()
    manager = _RegionalRunner(
        host, quality, doc, model="sdxl", cfg=cfg, seed=seed, device=device,
        asset_bytes=asset_bytes, loras=loras or NativeForgeLoRAs(),
        cancelled=cancelled, max_forwards_per_window=max_forwards_per_window,
        guidance=guidance, control_factory=control_factory,
        text_mode=text_mode)
    return execute_quality(
        quality, seed, manager.predict, device=device,
        inference_dtype=inference_dtype, cancelled=cancelled, progress=progress,
        cleanup=manager.close, max_cpu_bytes=max_cpu_bytes,
        max_window_bytes=max_window_bytes)


def run_forge_regional_anima(host, quality: AnimaQualityPlan, doc: dict, *,
                             cfg: float = 1.0, seed: int, asset_bytes=None,
                             loras=None, device=None, inference_dtype=None,
                             cancelled=None, progress=None,
                             max_cpu_bytes=4 * 1024**3,
                             max_window_bytes=512 * 1024**2,
                             max_forwards_per_window=16, guidance=(), control_factory=None):
    """Return P05 AnimaQualityResult; preserve 5D frozen state and RoPE."""
    if not isinstance(quality, AnimaQualityPlan):
        raise ValueError("AnimaQualityPlan required")
    if device is None:
        from backend import memory_management
        device = memory_management.get_torch_device()
    if inference_dtype is None:
        import torch
        model = getattr(getattr(getattr(host, "forge_objects", None), "unet", None), "model", None)
        inference_dtype = getattr(model, "computation_dtype", torch.float32)
    manager = _RegionalRunner(
        host, quality, doc, model="anima", cfg=cfg, seed=seed, device=device,
        asset_bytes=asset_bytes, loras=loras or NativeForgeLoRAs(),
        cancelled=cancelled, max_forwards_per_window=max_forwards_per_window,
        guidance=guidance, control_factory=control_factory)
    return execute_anima(
        quality, seed, manager.predict, device=device,
        inference_dtype=inference_dtype, cancelled=cancelled, progress=progress,
        cleanup=manager.close, max_cpu_bytes=max_cpu_bytes,
        max_window_bytes=max_window_bytes)
