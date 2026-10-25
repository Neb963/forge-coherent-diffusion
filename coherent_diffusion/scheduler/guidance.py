"""P06 typed, window-aligned structural guidance contract.

Maps are supplied locally as verified CPU image tensors. This layer never
downloads a model, guesses a ControlNet loader or mutates global host hooks.
The caller's Forge-native factory owns activation/pre-run/cleanup and must
fail on unsupported ControlNet/LLLite/Anima variants.
"""
from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from coherent_diffusion.crf.validation import validate_crf
from coherent_diffusion.scheduler.regional import UnsupportedRegional, _region_mask
from coherent_diffusion.scheduler.sdxl_sync import TilePlan


@dataclass(frozen=True)
class GuidanceRequest:
    name: str
    kind: str
    backend: str
    image: torch.Tensor  # [1,3,full_image_h,full_image_w] CPU, normalized [0,1]
    strength: float
    start: float
    end: float
    region_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GuidanceWindow:
    name: str
    kind: str
    backend: str
    image: torch.Tensor
    mask: torch.Tensor  # [1,1,crop_h,crop_w], 0..1 full-image coords
    strength: float


_ALLOWED = {
    "sdxl": {"sdxl_controlnet", "sdxl_lllite"},
    "anima": {"anima_vace", "anima_region_controlnet"},
}
_KINDS = {"color_layout", "lineart", "edge", "depth", "segmentation", "reference"}


def verified_guidance_png(raw: bytes, *, sha256: str, width: int, height: int):
    """Bounded PNG decode for a separately verified, supplied source image."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise UnsupportedRegional("Pillow required for structural source PNG") from exc
    if not isinstance(raw, bytes) or hashlib.sha256(raw).hexdigest() != sha256:
        raise UnsupportedRegional("Structural source hash mismatch")
    if type(width) is not int or type(height) is not int or min(width, height) < 1 or width*height > 1_048_576:
        raise UnsupportedRegional("Structural source pixel budget exceeded")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if im.format != "PNG" or im.size != (width, height):
                raise UnsupportedRegional("Structural source header/dimensions mismatch")
            rgb = im.convert("RGB")
            pixels = torch.tensor(bytearray(rgb.tobytes()), dtype=torch.uint8)
            return pixels.view(1, height, width, 3).permute(0, 3, 1, 2).contiguous().float()/255
    except (OSError, Image.DecompressionBombError) as exc:
        raise UnsupportedRegional("Unsafe structural source image") from exc


class GuidancePlan:
    def __init__(self, doc: dict, plan: TilePlan, model: str, requests=(), *,
                 asset_bytes=None, max_sources=4, max_pixels=1_048_576):
        validate_crf(doc, target_canvas=(plan.width_px, plan.height_px))
        if model not in _ALLOWED or not isinstance(plan, TilePlan):
            raise UnsupportedRegional("Unknown model/geometry for guidance")
        if doc["canvas"]["width"] * plan.height_px != doc["canvas"]["height"] * plan.width_px:
            raise UnsupportedRegional("Guidance aspect differs from generation aspect")
        if len(requests) > max_sources or plan.width_px*plan.height_px > max_pixels:
            raise UnsupportedRegional("Structural guidance budget exceeded")
        region_index = {r["id"]: r for r in doc["regions"]}
        masks = []
        names = set()
        for req in requests:
            if not isinstance(req, GuidanceRequest) or req.name in names:
                raise UnsupportedRegional("Unique typed GuidanceRequest required")
            names.add(req.name)
            if req.kind not in _KINDS or req.backend not in _ALLOWED[model]:
                raise UnsupportedRegional("Structural kind/backend unsupported by model")
            if (not all(isinstance(x, (float, int)) and math.isfinite(x)
                        for x in (req.strength, req.start, req.end))
                    or not (0 < req.strength <= 2 and 0 <= req.start < req.end <= 1)):
                raise UnsupportedRegional("Invalid structural strength/progress gate")
            image = req.image
            if (not isinstance(image, torch.Tensor) or image.dtype != torch.float32
                    or image.device.type != "cpu" or image.shape != (1, 3, plan.height_px, plan.width_px)
                    or not bool(torch.isfinite(image).all())
                    or bool(torch.any((image < 0) | (image > 1)))):
                raise UnsupportedRegional("Full-canvas RGB structural image must be finite CPU [0,1]")
            combined = torch.ones((plan.height, plan.width), dtype=torch.float32)
            if req.region_ids:
                combined.zero_()
                for rid in req.region_ids:
                    if rid not in region_index or not region_index[rid].get("enabled", True):
                        raise UnsupportedRegional("Structural guidance references missing/disabled region")
                    combined = torch.maximum(combined, _region_mask(
                        region_index[rid], doc, plan, asset_bytes=asset_bytes or {}))
            mask = F.interpolate(combined.view(1, 1, plan.height, plan.width),
                                 size=(plan.height_px, plan.width_px),
                                 mode="nearest")
            masks.append((req, image.detach().clone(), mask))
        self.geometry = plan
        self.requests = tuple(masks)
        self.model = model

    def for_window(self, tile, progress: float):
        if tile not in self.geometry.windows or not math.isfinite(progress) or not 0 <= progress <= 1:
            raise UnsupportedRegional("Invalid guidance window/progress")
        r = tile.read
        scale = self.geometry.scale
        yy = slice(r.y0*scale, r.y1*scale)
        xx = slice(r.x0*scale, r.x1*scale)
        return tuple(GuidanceWindow(req.name, req.kind, req.backend,
                                    image[:, :, yy, xx], mask[:, :, yy, xx],
                                    req.strength)
                     for req, image, mask in self.requests
                     if req.start <= progress <= req.end)
