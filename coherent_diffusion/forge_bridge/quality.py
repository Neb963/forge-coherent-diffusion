"""Opt-in P04 Forge SDXL quality bridge; no global callbacks or loader changes.

Routes both quality stages to P03's own source-backed ForgeSDXL adapter
with stage-correct prompt size metadata and explicit prepare/cleanup.
The full Forge/model/GPU path is not tested until P07.
"""
from __future__ import annotations

from coherent_diffusion.forge_bridge.sdxl import ForgeSDXL
from coherent_diffusion.scheduler.quality import QualityPlan, execute_quality


def run_forge_sdxl_quality(host, quality: QualityPlan, *, prompt: str,
                           negative_prompt: str = "", cfg: float, seed: int,
                           device=None, inference_dtype=None,
                           cancelled=None, progress=None,
                           max_cpu_bytes: int = 4 * 1024**3,
                           max_window_bytes: int = 512 * 1024**2):
    """Call the P03 SDXL denoiser sequentially at each P04 stage's dimensions.

    No implicit retile/retry is performed here; each stage's geometry is fixed
    and a failed run is rejected atomically. P03's bounded OOM replay is still
    available for standalone direct mode through ForgeSDXL.run().
    """
    if not isinstance(quality, QualityPlan):
        raise ValueError("Expected P04 QualityPlan")
    if device is None:
        from backend import memory_management
        device = memory_management.get_torch_device()
    active = None
    active_name = None

    def close():
        nonlocal active, active_name
        if active is not None:
            try:
                active.cleanup()
            finally:
                active = None
                active_name = None

    def predict(crop, sigma, tile, stage):
        nonlocal active, active_name
        from coherent_diffusion.forge_bridge.diagnostics import record, timed_prediction
        with timed_prediction(stage=stage, tile=tile.index, sigma=sigma):
            if active_name != stage:
                close()
                selected = quality.anchor if stage == "anchor" else quality.target
                if selected is None:
                    raise ValueError("Requested missing anchor stage")
                record("conditioning_begin", stage=stage, tile=tile.index)
                candidate = ForgeSDXL.from_prompts(
                    host, selected, prompt=prompt, negative_prompt=negative_prompt,
                    cfg=cfg, seed=seed, device=device)
                record("conditioning_done", stage=stage, tile=tile.index)
                record("model_prepare_begin", stage=stage, tile=tile.index)
                candidate.prepare()
                record("model_prepare_done", stage=stage, tile=tile.index)
                active = candidate
                active_name = stage
            return active.predict(crop, sigma, tile)

    return execute_quality(
        quality, seed, predict, device=device,
        inference_dtype=inference_dtype, cancelled=cancelled,
        progress=progress, cleanup=close, max_cpu_bytes=max_cpu_bytes,
        max_window_bytes=max_window_bytes)
