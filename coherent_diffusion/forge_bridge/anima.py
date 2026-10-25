"""P05 opt-in Forge Neo Anima source-contract bridge (not a UI sampler hook).

The user's installed Q4 GGUF/Qwen encoder is invoked via the existing host
get_learned_conditioning, not replaced, wrapped in a different loader, cloned
or quantized. Installed runtime / GGUF correctness is reserved for P07.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from coherent_diffusion.forge_bridge.anima_position import scoped_absolute_rope
from coherent_diffusion.scheduler.anima_flow import (
    AnimaQualityPlan, execute_anima, validate_anima_tiles,
)


class UnsupportedAnima(RuntimeError):
    pass


class _PromptBatch(list):
    def __init__(self, text: str, negative: bool):
        super().__init__([text])
        self.is_negative_prompt = negative


def _check_unet_hooks(unet):
    """Reject unowned sampling hooks, with a precise non-mutating diagnosis.

    A bare empty transformer_options dict is an inert bookkeeping container
    used by extension APIs, not an active attention hook. All other nonempty
    options still require ownership negotiation (P07 live gate).
    """
    if getattr(unet, "controlnet_linked_list", None) is not None:
        raise UnsupportedAnima(
            "Active UNet ControlNet attachment; native tiled ControlNet integration "
            "is not yet verified. Disable that ControlNet or use native txt2img.")
    if getattr(unet, "extra_concat_condition", None) is not None:
        raise UnsupportedAnima(
            "Active UNet extra_concat_condition; external/inpaint conditioning "
            "cannot yet be tiled safely.")
    options = getattr(unet, "model_options", None)
    if options == {} or options == {"transformer_options": {}}:
        return
    if not isinstance(options, dict):
        raise UnsupportedAnima("UNet model_options is unavailable or not a dict")
    nested = options.get("transformer_options")
    if set(options) == {"transformer_options"} and isinstance(nested, dict) and set(nested) == {"sampling_sigmas"}:
        # A native/extension sampler can leave its scalar sigma timeline in
        # transformer_options after normal txt2img. This is numerical schedule
        # metadata, not a callable hook or a ControlNet attachment. P07 runs
        # its own independently derived predictor sigmas; do not reuse these
        # values or mutate the host model_options. Require a bounded finite
        # 1-D floating sequence and accept no additional option keys.
        import torch
        timeline = nested["sampling_sigmas"]
        if isinstance(timeline, torch.Tensor):
            valid = (timeline.ndim == 1 and 2 <= timeline.numel() <= 1024
                     and torch.is_floating_point(timeline)
                     and bool(torch.isfinite(timeline).all()))
        elif isinstance(timeline, (tuple, list)):
            valid = (2 <= len(timeline) <= 1024
                     and all(type(v) in (float, int) and math.isfinite(v)
                             for v in timeline))
        else:
            valid = False
        if not valid:
            raise UnsupportedAnima(
                "UNet transformer_options.sampling_sigmas has an invalid "
                "type, shape, length or nonfinite value")
        return
    keys = sorted(str(k)[:64] for k in options)[:24]
    extra = " (+ more)" if len(options) > len(keys) else ""
    nested_keys = (sorted(str(k)[:64] for k in nested)[:12]
                   if isinstance(nested, dict) else [])
    raise UnsupportedAnima(
        f"Unnegotiated UNet model_options keys: {keys}{extra}; "
        f"transformer_option_keys: {nested_keys}. "
        "Active or unknown extension hooks cannot be silently discarded.")

@dataclass
class ForgeAnima:
    host: object
    plan: object
    positive: object
    negative: object
    cfg: float
    seed: int
    device: object
    patch_spatial: int
    _prepared: bool = False

    @staticmethod
    def check_host(host):
        """Class/shape/prediction/position gates; no model or loader mutation."""
        if (type(host).__name__ != "Anima" or getattr(host, "is_wan", False) is not True
                or getattr(host, "is_sdxl", False) is True
                or getattr(host, "is_inpaint", True)):
            raise UnsupportedAnima("Requires Forge Neo Anima one-frame non-inpaint model")
        obj = getattr(host, "forge_objects", None)
        unet, vae, clip = (getattr(obj, n, None) for n in ("unet", "vae", "clip"))
        model = getattr(unet, "model", None)
        core = getattr(model, "diffusion_model", None)
        predictor = getattr(model, "predictor", None)
        if (unet is None or vae is None or clip is None
                or not callable(getattr(model, "apply_model", None))
                or type(core).__name__ != "Anima"
                or type(core).__module__ != "backend.nn.anima"
                or type(predictor).__name__ not in ("PredictionFlux", "PredictionDiscreteFlow")
                or getattr(predictor, "prediction_type", None) != "const"):
            raise UnsupportedAnima("Unrecognized native Anima DiT/constant-flow predictor")
        patch = getattr(core, "patch_spatial", None)
        if (type(patch) is not int or patch < 1 or patch > 16
                or getattr(core, "patch_temporal", None) != 1
                or getattr(core, "in_channels", None) != 16
                or getattr(core, "out_channels", None) != 16
                or type(getattr(core, "pos_embedder", None)).__name__ != "VideoRopePosition3DEmb"):
            raise UnsupportedAnima("Unsupported DiT patches, channels or RoPE implementation")
        if (getattr(vae, "is_wan", False) is not True
                or getattr(vae, "latent_dim", None) != 3
                or getattr(vae, "latent_channels", None) != 16
                or not isinstance(getattr(vae, "upscale_ratio", None), tuple)
                or vae.upscale_ratio[1:] != (8, 8)
                or not callable(getattr(vae, "decode_tiled", None))
                or not callable(getattr(getattr(vae, "first_stage_model", None), "process_out", None))):
            raise UnsupportedAnima("Requires native WAN 3D 16-channel tiled VAE")
        _check_unet_hooks(unet)
        if (getattr(host, "ini_latent", None) is not None
                or bool(getattr(host, "ref_latents", []))):
            raise UnsupportedAnima("Anima reference-image concatenation unsupported by tiled RoPE")
        if not callable(getattr(host, "get_learned_conditioning", None)):
            raise UnsupportedAnima("Existing Anima text encoder entrypoint missing")
        # The live host's global dynamic reference list is another reference path.
        try:
            from backend.args import dynamic_args
            if getattr(dynamic_args, "ref_latents", None):
                raise UnsupportedAnima("Active Forge reference latents unsupported")
        except ImportError:
            pass
        return patch

    @classmethod
    def from_prompts(cls, host, plan, *, prompt: str, negative_prompt: str = "",
                     cfg: float = 1.0, seed: int, device=None):
        patch = cls.check_host(host)
        validate_anima_tiles(plan, patch_spatial=patch)
        if not isinstance(prompt, str) or not isinstance(negative_prompt, str):
            raise ValueError("Anima prompts must be strings")
        if not isinstance(cfg, (int, float)) or not math.isfinite(cfg) or cfg < 0:
            raise ValueError("Finite nonnegative CFG required")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Explicit nonnegative signed Anima seed required")
        if device is None:
            from backend import memory_management
            device = memory_management.get_torch_device()
        import torch
        from backend.sampling.condition import compile_conditions
        # One bounded encoder call per unique prompt polarity and adapter run.
        # Never clone/reload/requantize the installed Q4 or alternative encoder.
        def encode(text, negative):
            value = host.get_learned_conditioning(_PromptBatch(text, negative))
            if not isinstance(value, (list, tuple)) or len(value) != 1:
                raise UnsupportedAnima("Unexpected Anima Qwen embedding contract")
            tensor = value[0]
            if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 3
                    or tensor.shape[0] != 1 or tensor.shape[1] < 1
                    or not bool(torch.isfinite(tensor).all())):
                raise UnsupportedAnima("Invalid installed Anima text embedding")
            return compile_conditions(tensor)
        positive = encode(prompt, False)
        negative = encode(negative_prompt, True)
        return cls(host, plan, positive, negative, float(cfg), seed, device, patch)

    def prepare(self):
        if self._prepared:
            raise RuntimeError("Anima adapter is already prepared")
        self.check_host(self.host)
        validate_anima_tiles(self.plan, patch_spatial=self.patch_spatial)
        import torch
        from backend.sampling.sampling_function import sampling_prepare, sampling_cleanup
        tile = max(self.plan.windows, key=lambda w: w.read.width * w.read.height)
        # The source's sampling_prepare reads shapes for GPU memory reservation.
        shape = torch.empty((1, 16, 1, tile.read.height, tile.read.width), device="meta")
        try:
            sampling_prepare(self.host.forge_objects.unet, shape)
        except BaseException:
            sampling_cleanup(self.host.forge_objects.unet)
            raise
        self._prepared = True

    def predict(self, crop, sigma, tile):
        if not self._prepared:
            raise RuntimeError("Anima adapter must be prepared")
        self.check_host(self.host)
        import torch
        from backend.sampling.sampling_function import sampling_function_inner
        model = self.host.forge_objects.unet.model
        core = model.diffusion_model
        s = torch.full((crop.shape[0],), sigma, device=crop.device, dtype=crop.dtype)
        # We only temporarily alter this DiT instance while our caller owns the
        # model exclusively; native KModel continues to convert flow to x0.
        with scoped_absolute_rope(core, tile, patch_spatial=self.patch_spatial):
            return sampling_function_inner(
                model, crop, s, self.negative, self.positive, self.cfg, {}, self.seed)

    def cleanup(self):
        if self._prepared:
            self._prepared = False
            from backend.sampling.sampling_function import sampling_cleanup
            sampling_cleanup(self.host.forge_objects.unet)

    def decode_tiled(self, latent, *, tile_latent: int = 32, overlap_latent: int = 8):
        import torch
        self.check_host(self.host)
        if (not isinstance(latent, torch.Tensor) or latent.ndim != 5
                or latent.device.type != "cpu" or latent.shape[1] != 16
                or latent.shape[2] != 1 or not bool(torch.isfinite(latent).all())):
            raise ValueError("WAN VAE decode requires finite CPU [B,16,1,H,W]")
        if (type(tile_latent) is not int or type(overlap_latent) is not int
                or tile_latent < 2 or overlap_latent < 0 or overlap_latent >= tile_latent):
            raise ValueError("Invalid native WAN decoder tile settings")
        vae = self.host.forge_objects.vae
        # WAN decoder runs its 3D tiled implementation; do not reuse SDXL 2D.
        scaled = vae.first_stage_model.process_out(latent)
        output = vae.decode_tiled(scaled, tile_x=tile_latent,
                                  tile_y=tile_latent, overlap=overlap_latent)
        if (not isinstance(output, torch.Tensor) or output.ndim != 5
                or output.shape[1] != 1 or output.shape[-1] != 3):
            raise UnsupportedAnima("Native WAN decoder output layout unsupported")
        return output[:, 0]


def run_forge_anima_quality(host, plan: AnimaQualityPlan, *, prompt: str,
                            negative_prompt: str = "", cfg: float = 1.0,
                            seed: int, device=None, inference_dtype=None,
                            cancelled=None, progress=None,
                            max_cpu_bytes: int = 4 * 1024**3,
                            max_window_bytes: int = 512 * 1024**2):
    """Two flow-native modes; reuses one loaded host and its existing text route."""
    if not isinstance(plan, AnimaQualityPlan):
        raise ValueError("Expected Anima quality plan")
    if device is None:
        from backend import memory_management
        device = memory_management.get_torch_device()
    active = None
    stage_name = None

    def close():
        nonlocal active, stage_name
        if active is not None:
            try:
                active.cleanup()
            finally:
                stage_name = None

    def predict(crop, sigma, tile, stage):
        nonlocal active, stage_name
        from coherent_diffusion.forge_bridge.diagnostics import record, timed_prediction
        with timed_prediction(stage=stage, tile=tile.index, sigma=sigma):
            if active is None:
                record("conditioning_begin", stage=stage, tile=tile.index)
                active = ForgeAnima.from_prompts(
                    host, plan.target, prompt=prompt, negative_prompt=negative_prompt,
                    cfg=cfg, seed=seed, device=device)
                record("conditioning_done", stage=stage, tile=tile.index)
            if stage_name != stage:
                close()
                active.plan = plan.anchor if stage == "anchor" else plan.target
                record("model_prepare_begin", stage=stage, tile=tile.index)
                active.prepare()
                record("model_prepare_done", stage=stage, tile=tile.index)
                stage_name = stage
            return active.predict(crop, sigma, tile)

    import torch
    if inference_dtype is None:
        model = getattr(getattr(host, "forge_objects", None), "unet", None)
        inference_dtype = getattr(getattr(model, "model", None), "computation_dtype",
                                  torch.float32)
    return execute_anima(
        plan, seed, predict, device=device,
        inference_dtype=inference_dtype, cancelled=cancelled,
        progress=progress, cleanup=close,
        max_cpu_bytes=max_cpu_bytes, max_window_bytes=max_window_bytes)
