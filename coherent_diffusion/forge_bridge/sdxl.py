"""Source-backed Forge Neo SDXL P03 adapter.

Opt-in Python API only. Does not register generation controls or replace Forge's
sampler/CLIP/VAE/LoRA objects. Host-aware errors are explicit. P07 verifies real
Forge Neo and actual SDXL checkpoints; this source contract alone cannot.
"""
from __future__ import annotations

from dataclasses import dataclass

from coherent_diffusion.scheduler.sdxl_sync import TilePlan, seeded_noise, synchronized_euler


class UnsupportedSDXL(RuntimeError):
    pass


def _host_oom(exc):
    """Recognize Forge's own OOM classification without string heuristics."""
    import torch
    if isinstance(exc, (MemoryError, torch.cuda.OutOfMemoryError)):
        return True
    try:
        from backend import memory_management
        return bool(memory_management.is_oom(exc))
    except (ImportError, AttributeError):
        return False


class _PromptBatch(list):
    """SDXL's existing get_learned_conditioning reads these request metadata."""
    def __init__(self, prompt: str, width: int, height: int, negative: bool):
        super().__init__([prompt])
        self.width = width
        self.height = height
        self.is_negative_prompt = negative


@dataclass
class ForgeSDXL:
    host: object
    plan: TilePlan
    positive: object
    negative: object
    cfg: float
    seed: int
    device: object
    _prepared: bool = False

    @classmethod
    def from_prompts(cls, host, plan: TilePlan, *, prompt: str, negative_prompt: str,
                     cfg: float, seed: int, device=None):
        """Compile existing Forge conditioning once; no text-encoder replacement."""
        cls._check_host(host)
        if not isinstance(prompt, str) or not isinstance(negative_prompt, str):
            raise ValueError("SDXL prompt and negative prompt must be strings")
        import math
        if not isinstance(cfg, (int, float)) or not math.isfinite(cfg) or cfg < 0:
            raise ValueError("Finite nonnegative CFG required")
        if type(seed) is not int:
            raise ValueError("Integer seed required")
        if device is None:
            from backend import memory_management
            device = memory_management.get_torch_device()
        from backend.sampling.condition import compile_conditions
        p = host.get_learned_conditioning(_PromptBatch(prompt, plan.width_px, plan.height_px, False))
        n = host.get_learned_conditioning(_PromptBatch(negative_prompt, plan.width_px, plan.height_px, True))
        if not isinstance(p, dict) or not isinstance(n, dict):
            raise UnsupportedSDXL("SDXL conditioning dictionary unavailable")
        return cls(host, plan, compile_conditions(p), compile_conditions(n), float(cfg), seed, device)

    @staticmethod
    def _check_host(host):
        if type(host).__name__ != "StableDiffusionXL" or getattr(host, "is_sdxl", False) is not True:
            raise UnsupportedSDXL("P03 supports Forge StableDiffusionXL base only; not Anima/refiner")
        if getattr(host, "is_inpaint", True) or getattr(host, "use_shift", True):
            raise UnsupportedSDXL("SDXL inpaint or rectified-flow model not supported by Euler P03")
        obj = getattr(host, "forge_objects", None)
        unet = getattr(obj, "unet", None)
        vae = getattr(obj, "vae", None)
        if unet is None or vae is None or not callable(getattr(vae, "decode_tiled", None)):
            raise UnsupportedSDXL("Missing Forge U-Net or tiled VAE API")
        if getattr(unet, "controlnet_linked_list", None) is not None:
            raise UnsupportedSDXL("ControlNet/LLLite per-tile spatial remapping is P06 only")
        if getattr(unet, "extra_concat_condition", None) is not None:
            raise UnsupportedSDXL("Concat/inpaint conditioning not implemented by P03")
        options = getattr(unet, "model_options", {})
        if not isinstance(options, dict) or options:
            raise UnsupportedSDXL("Nonempty Forge model options/hook state unsupported by P03")
        if getattr(vae, "upscale_ratio", None) != 8 or getattr(vae, "latent_channels", None) != 4:
            raise UnsupportedSDXL("Requires SDXL 4-channel, 8x VAE geometry")
        if not callable(getattr(host, "get_learned_conditioning", None)):
            raise UnsupportedSDXL("Forge SDXL encoder entrypoint unavailable")
        if getattr(unet, "model", None) is None:
            raise UnsupportedSDXL("Forge U-Net patcher lacks model")
        if not callable(getattr(unet.model, "apply_model", None)):
            raise UnsupportedSDXL("Forge diffusion model apply_model unavailable")

    def prepare(self):
        if self._prepared:
            raise RuntimeError("SDXL adapter already prepared")
        self._check_host(self.host)
        import torch
        from backend.sampling.sampling_function import sampling_prepare
        unet = self.host.forge_objects.unet
        largest = max(self.plan.windows, key=lambda t: t.read.width * t.read.height)
        # Host source inspects shape to reserve activation memory; does not
        # imply a full-image GPU allocation. Allocate on CPU only.
        sample = torch.empty((1, 4, largest.read.height, largest.read.width), device="meta")
        try:
            sampling_prepare(unet, sample)
        except BaseException:
            # A host prepare may have partially loaded/registered resources.
            from backend.sampling.sampling_function import sampling_cleanup
            sampling_cleanup(unet)
            raise
        self._prepared = True

    def predict(self, crop, sigma: float, tile):
        if not self._prepared:
            raise RuntimeError("SDXL adapter must be prepared before prediction")
        import torch
        from backend.sampling.sampling_function import sampling_function_inner
        model = self.host.forge_objects.unet.model
        s = torch.full((crop.shape[0],), sigma, dtype=crop.dtype, device=crop.device)
        result = sampling_function_inner(model, crop, s, self.negative, self.positive,
                                         self.cfg, {}, self.seed)
        return result

    def cleanup(self):
        if self._prepared:
            self._prepared = False
            from backend.sampling.sampling_function import sampling_cleanup
            sampling_cleanup(self.host.forge_objects.unet)

    def run(self, sigmas, *, max_cpu_bytes=4 * 1024**3, max_window_bytes=512 * 1024**2,
            cancelled=None, progress=None, inference_dtype=None,
            min_tile_px=256, max_retiles=2):
        """Bounded OOM recovery replays the ENTIRE run with the same global seed.

        Tile shrink never resumes mid-step or keeps partial solver history. Only
        MemoryError/CUDA OOM retries; all other errors fail without rerouting.
        Caller can inspect result.plan to see actual geometry. Does not claim
        any measured device VRAM requirement.
        """
        import torch
        from coherent_diffusion.scheduler.sdxl_sync import _validate_schedule
        schedule = _validate_schedule(sigmas)
        if (type(min_tile_px) is not int or type(max_retiles) is not int
                or min_tile_px < self.plan.scale or min_tile_px % self.plan.scale
                or max_retiles < 0 or max_retiles > 4):
            raise ValueError("Invalid bounded retry settings")
        if type(max_cpu_bytes) is not int or type(max_window_bytes) is not int:
            raise ValueError("Integer memory budgets required")
        if inference_dtype is None:
            # Forge records the model's actual inference precision here.
            inference_dtype = getattr(self.host.forge_objects.unet.model,
                                      "computation_dtype", torch.float32)
        for attempt in range(max_retiles + 1):
            p = self.plan
            # Check global residency budget BEFORE allocating noise or loading the model.
            master_bytes = p.width * p.height * 4 * 4  # batch=1, channels=4, fp32
            if 4 * master_bytes + p.width * p.height * 4 > max_cpu_bytes:
                raise MemoryError("CPU global field exceeds configured residency budget")
            max_crop_bytes = max(4 * t.read.width * t.read.height * 4 for t in p.windows)
            try:
                if max_crop_bytes > max_window_bytes:
                    raise MemoryError("Inference window exceeds configured tile budget")
                if cancelled is not None and cancelled():
                    raise InterruptedError("Coherent tiled run interrupted")
                self.prepare()
                try:
                    initial = seeded_noise(p, self.seed, sigma=schedule[0])
                except BaseException:
                    self.cleanup()
                    raise
                return synchronized_euler(
                    p, initial, schedule, self.predict, device=self.device,
                    inference_dtype=inference_dtype,
                    cancelled=cancelled, progress=progress, cleanup=self.cleanup,
                    max_cpu_bytes=max_cpu_bytes, max_window_bytes=max_window_bytes)
            except Exception as error:
                if not _host_oom(error):
                    raise
                self.cleanup()
                next_tile = max(min_tile_px, (p.tile_px // (2 * p.scale)) * p.scale)
                if attempt >= max_retiles or next_tile >= p.tile_px:
                    raise MemoryError(
                        "SDXL tiled P03 exhausted bounded memory retries; "
                        "reduce tile size/resolution or release GPU memory") from error
                next_overlap = min(p.overlap_px, (next_tile // (4 * p.scale)) * p.scale)
                next_halo = min(p.halo_px, (next_tile // (4 * p.scale)) * p.scale)
                self.plan = TilePlan.build(p.width_px, p.height_px, next_tile,
                                           next_overlap, next_halo, scale=p.scale)

    def decode_tiled(self, latent, *, tile_latent=64, overlap_latent=16):
        """Preserve model's own latent scaling and Forge tiled VAE; return NHWC."""
        self._check_host(self.host)
        import torch
        if not isinstance(latent, torch.Tensor) or latent.device.type != "cpu":
            raise ValueError("Tiled decode requires CPU latent")
        vae = self.host.forge_objects.vae
        scaled = vae.first_stage_model.process_out(latent)
        return vae.decode_tiled(scaled, tile_x=tile_latent,
                                tile_y=tile_latent, overlap=overlap_latent)
