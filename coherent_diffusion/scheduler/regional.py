"""P06 CRF-to-latent regional coefficients and grouped frozen-state x0 blending.

Pure opt-in tensor execution. One mask field in global latent coordinates; no
per-tile independent sampler and no claims about actual visual attribution.
"""
from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from coherent_diffusion.crf.validation import validate_crf
from coherent_diffusion.scheduler.sdxl_sync import TilePlan


class UnsupportedRegional(ValueError):
    """Explicit unsupported/unsafe authoring or host capabilities."""


@dataclass(frozen=True)
class PredictionState:
    prompt: str
    negative: str
    adapters: tuple[tuple[str, str | None, float], ...] = ()


@dataclass(frozen=True)
class RegionWeights:
    state: PredictionState
    weight: torch.Tensor  # [H,W] float32 CPU, nonnegative


@dataclass(frozen=True)
class RegionalPlan:
    geometry: TilePlan
    states: tuple[RegionWeights, ...]
    region_ids: tuple[str, ...]
    warnings: tuple[str, ...]
    max_forwards_per_window: int

    def window(self, tile):
        """Return only positive-coverage state contributions for a halo read."""
        if tile not in self.geometry.windows:
            raise UnsupportedRegional("Window does not belong to regional geometry")
        r = tile.read
        return tuple((s.state, s.weight[r.y0:r.y1, r.x0:r.x1])
                     for s in self.states
                     if bool(torch.any(s.weight[r.y0:r.y1, r.x0:r.x1] > 0)))


def _mask_image(meta, raw, size):
    """Decode verified PNG to bounded grayscale mask; no external paths or fetching."""
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise UnsupportedRegional("Pillow required for verified generation masks") from exc
    if not isinstance(raw, bytes) or hashlib.sha256(raw).hexdigest() != meta["sha256"]:
        raise UnsupportedRegional("Mask bytes/hash missing or invalid")
    if meta["media_type"] != "image/png":
        raise UnsupportedRegional("WebP masks cannot be decoded by the verified PNG path")
    try:
        with Image.open(io.BytesIO(raw)) as src:
            if src.format != "PNG" or src.size != (meta["width"], meta["height"]):
                raise UnsupportedRegional("Mask MIME/header/dimensions disagree with CRF")
            if src.width * src.height > 16_777_216:
                raise UnsupportedRegional("Mask decoder pixel budget exceeded")
            if src.mode in ("RGBA", "LA"):
                channel = src.getchannel("A")
            elif src.mode == "P" and "transparency" in src.info:
                channel = src.convert("RGBA").getchannel("A")
            else:
                channel = src.convert("L")
            data = torch.tensor(bytearray(channel.tobytes()), dtype=torch.uint8)
            data = data.view(1, 1, src.height, src.width).to(torch.float32) / 255.0
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise UnsupportedRegional("Invalid or oversized PNG mask pixels") from exc
    mode = size[2]
    return F.interpolate(data, size=size[:2], mode="bilinear" if mode == "bilinear" else "nearest").view(*size[:2]).clamp_(0, 1)


def _signed_geometry(geom, plan):
    """Signed distance in OUTPUT pixels at global latent centers; positive inside."""
    h, w = plan.height, plan.width
    y = (torch.arange(h, dtype=torch.float32) + 0.5) * plan.scale
    x = (torch.arange(w, dtype=torch.float32) + 0.5) * plan.scale
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    if geom["type"] == "bbox":
        x0, y0, bw, bh = geom["bbox"]
        left, top = x0 * plan.width_px, y0 * plan.height_px
        right, bottom = (x0 + bw) * plan.width_px, (y0 + bh) * plan.height_px
        inside = torch.minimum(torch.minimum(xx-left, right-xx),
                               torch.minimum(yy-top, bottom-yy))
        return inside
    points = [(p[0]*plan.width_px, p[1]*plan.height_px) for p in geom["points"]]
    inside = torch.zeros((h, w), dtype=torch.bool)
    distance_sq = torch.full((h, w), float("inf"), dtype=torch.float32)
    for a, b in zip(points, points[1:] + points[:1]):
        ax, ay = a
        bx, by = b
        dx, dy = bx-ax, by-ay
        segment = dx*dx + dy*dy
        t = ((xx-ax)*dx + (yy-ay)*dy) / segment
        t = t.clamp(0, 1)
        d = (xx-(ax+t*dx)).square() + (yy-(ay+t*dy)).square()
        distance_sq = torch.minimum(distance_sq, d)
        crosses = ((ay > yy) != (by > yy)) & (
            xx < (bx-ax) * (yy-ay) / (by-ay if abs(by-ay) > 1e-12 else 1e-12) + ax)
        inside ^= crosses
    dist = distance_sq.sqrt()
    return torch.where(inside, dist, -dist)


def _feather_bitmap(mask: torch.Tensor, feather_px: float, scale: int) -> torch.Tensor:
    """Separable O(HW) box blur, replicate edge convention, width in image pixels.

    Explicitly approximates signed-distance vector feather; fractional bitmap
    alpha remains meaningful. Never create large 2D convolution kernels.
    """
    if feather_px <= 0:
        return mask
    radius = max(1, int(round(feather_px / (2.0 * scale))))
    radius = min(radius, 256)
    width = radius * 2 + 1
    x = mask.reshape(1, 1, *mask.shape)
    horizontal = F.pad(x, (radius, radius, 0, 0), mode="replicate")
    csum = F.pad(horizontal.cumsum(dim=-1), (1, 0, 0, 0))
    x = (csum[..., width:] - csum[..., :-width]) / width
    vertical = F.pad(x, (0, 0, radius, radius), mode="replicate")
    csum_y = F.pad(vertical.cumsum(dim=-2), (0, 0, 1, 0))
    x = (csum_y[..., width:, :] - csum_y[..., :-width, :]) / width
    return x.view_as(mask).clamp_(0, 1)


def _region_mask(region, doc, plan, *, asset_bytes):
    geom = region["geometry"]
    if geom["type"] == "mask":
        meta = next((x for x in doc.get("assets", []) if x["id"] == geom["asset_id"]), None)
        if meta is None or geom["asset_id"] not in asset_bytes:
            raise UnsupportedRegional("Mask requires hash-verified archive bytes")
        mask = _mask_image(meta, asset_bytes[geom["asset_id"]],
                           (plan.height, plan.width, geom.get("interpolation", "nearest")))
        return _feather_bitmap(mask, region["blend"]["feather_px"], plan.scale)
    signed = _signed_geometry(geom, plan)
    feather = region["blend"]["feather_px"]
    if feather == 0:
        return (signed >= 0).float()
    return (signed / feather + 0.5).clamp(0, 1)


def _entity_prompt(region_id, doc):
    return ", ".join(dict.fromkeys(
        e["description"].strip() for e in doc.get("entities", [])
        if region_id in e["region_ids"] and e["description"].strip()))


def _region_state(region, doc):
    style = doc["global"]["style_prompt"] if region.get("inherit_style", True) else ""
    value = region.get("prompt")
    # Missing/null inherits shared style ONLY; empty explicitly yields no local text.
    # Null/missing local text may refer to declared unique entities; a
    # deliberately empty string suppresses entity fallback.
    if value is None:
        value = _entity_prompt(region["id"], doc)
    parts = [x.strip() for x in (style, value) if isinstance(x, str) and x.strip()]
    prompt = ", ".join(dict.fromkeys(parts))
    neg = region.get("negative_prompt")
    if neg is None:
        neg = doc["global"].get("negative_prompt") or ""
    adapters = []
    for a in region.get("loras", []):
        if a["requested_isolation"] == "experimental_internal":
            raise UnsupportedRegional("Internal adapter residual masking is experimental/disabled")
        if a.get("te_weight") not in (None, 0):
            raise UnsupportedRegional("Regional text-encoder LoRAs cannot safely modify a shared/Q4 encoder")
        adapters.append((a["name"], a.get("hash"), float(a["weight"])))
    # Same adapter set in different authoring order is one model state.
    return PredictionState(prompt, neg, tuple(sorted(adapters)))


def compile_regional(doc: dict, plan: TilePlan, *, asset_bytes=None,
                     max_latent_pixels=None, max_forwards_per_window=None,
                     include_scene_in_base=False,
                     max_mask_workspace_bytes=1024**3):
    """Build complete normalized per-state coefficients, no I/O/model mutations.

    Supports exact normalized aspect only; projective/crop transformations need
    explicit preconverted CRF. Window extraction uses the global latent grid.
    """
    if not isinstance(plan, TilePlan):
        raise ValueError("TilePlan required")
    validate_crf(doc, target_canvas=(plan.width_px, plan.height_px))
    c = doc["canvas"]
    if c["width"] * plan.height_px != c["height"] * plan.width_px:
        raise UnsupportedRegional("Aspect mismatch requires explicit reproject before generation")
    if (max_latent_pixels is not None and
            (type(max_latent_pixels) is not int or max_latent_pixels < 1 or
             plan.width * plan.height > max_latent_pixels)):
        raise UnsupportedRegional("Explicit regional latent allowance exceeded")
    if max_forwards_per_window is not None and (
            type(max_forwards_per_window) is not int or max_forwards_per_window < 1):
        raise UnsupportedRegional("Invalid regional forward-work allowance")
    assets = asset_bytes if asset_bytes is not None else {}
    if not isinstance(assets, dict):
        raise UnsupportedRegional("Verified mask bytes must be keyed by CRF asset ID")
    base = float(doc["global"]["base_weight"])
    neg = doc["global"].get("negative_prompt") or ""
    # The local baseline is shared style, not repeated scene/object declarations.
    if type(include_scene_in_base) is not bool:
        raise UnsupportedRegional("Explicit direct-scene baseline flag required")
    base_prompt = doc["global"]["style_prompt"]
    if include_scene_in_base:
        base_prompt = ", ".join(dict.fromkeys(x.strip() for x in
            (doc["global"]["scene_prompt"], base_prompt) if x.strip()))
    initial = PredictionState(base_prompt, neg)
    # Conservative CPU mask/intermediate estimate before any per-region grid
    # construction. Geometry distance tensors need multiple full-size grids.
    active_regions = sum(bool(r.get("enabled", True) and r["blend"]["weight"] > 0)
                         for r in doc["regions"])
    if (type(max_mask_workspace_bytes) is not int or max_mask_workspace_bytes <= 0
            or (active_regions * 16 + 64) * plan.width * plan.height >
            max_mask_workspace_bytes):
        raise MemoryError("Regional mask compilation exceeds configured CPU workspace budget")
    masks = []
    warnings = []
    for index, region in enumerate(doc["regions"]):
        if not region.get("enabled", True) or region["blend"]["weight"] == 0:
            continue
        m = _region_mask(region, doc, plan, asset_bytes=assets)
        if not bool(torch.any(m > 0)):
            warnings.append(region["id"] + " has no covered latent sample")
            continue
        masks.append((index, region, m))
    weights = []
    if doc["blend"]["policy"] == "weighted":
        weights = [(r, m * r["blend"]["weight"]) for _, r, m in masks]
        base_weight = torch.full((plan.height, plan.width), base)
    elif doc["blend"]["policy"] == "priority":
        transmission = torch.ones((plan.height, plan.width), dtype=torch.float32)
        # Last authoring item wins ties, stable across window traversal.
        for _, r, m in sorted(masks, key=lambda entry:
                              (entry[1]["blend"]["z_index"], entry[0]), reverse=True):
            weights.append((r, transmission * m * r["blend"]["weight"]))
            transmission *= (1-m)
        # Match P02 authoring preview: base contribution is explicit and
        # is never occluded by higher-priority regional masks.
        base_weight = torch.full((plan.height, plan.width), base)
    else:
        raise UnsupportedRegional("Unsupported regional blend policy")
    denom = base_weight.clone()
    for _, w in weights:
        denom += w
    if bool(torch.any(denom <= 0)) or not bool(torch.isfinite(denom).all()):
        raise UnsupportedRegional("Region/base coefficients leave uncovered latent positions")
    grouped = {initial: base_weight / denom}
    for region, w in weights:
        state = _region_state(region, doc)
        if state in grouped:
            grouped[state] = grouped[state] + w / denom
        else:
            grouped[state] = w / denom
    result = tuple(RegionWeights(state, value.contiguous()) for state, value in grouped.items()
                   if bool(torch.any(value > 0)))
    if max_forwards_per_window is not None and len(result) > max_forwards_per_window:
        raise UnsupportedRegional("Distinct regional prediction states exceed explicit forward allowance")
    return RegionalPlan(plan, result, tuple(r["id"] for _, r, _ in masks),
                        tuple(warnings), max_forwards_per_window or len(result))


def blended_prediction(regional: RegionalPlan, tile, crop: torch.Tensor, sigma: float,
                       stage: str, predict: Callable, *, cancelled=None):
    """Blend only predictions from identical frozen crop/sigma, never tile final images.

    Caller serializes model/LoRA state transitions and returns denoised x0.
    Works for SDXL [B,4,H,W] and Anima [B,16,1,H,W] without changing native RoPE.
    """
    if stage == "anchor":
        raise UnsupportedRegional("Global scene anchor is supplied by the caller, not regional mixing")
    if crop.ndim not in (4, 5) or (crop.ndim == 5 and (crop.shape[1] != 16 or crop.shape[2] != 1)):
        raise UnsupportedRegional("SDXL 4D or one-frame Anima 5D crop required")
    if not callable(predict) or not isinstance(sigma, (int, float)) or not math.isfinite(sigma):
        raise UnsupportedRegional("Finite sigma/predictor required")
    expected = regional.window(tile)
    if not expected or len(expected) > regional.max_forwards_per_window:
        raise UnsupportedRegional("Uncovered or over-budget prediction window")
    out = torch.zeros_like(crop)
    for state, mask in expected:
        if cancelled is not None and cancelled():
            raise InterruptedError("Regional prediction interrupted")
        pred = predict(state, crop, sigma, tile, stage)
        if not isinstance(pred, torch.Tensor) or pred.shape != crop.shape:
            raise UnsupportedRegional("Regional predictor returned invalid shape")
        if not bool(torch.isfinite(pred).all()):
            raise UnsupportedRegional("Nonfinite regional prediction")
        if crop.ndim == 5:
            w = mask.to(device=crop.device, dtype=crop.dtype).view(1, 1, 1, *mask.shape)
        else:
            w = mask.to(device=crop.device, dtype=crop.dtype).view(1, 1, *mask.shape)
        out += pred * w
    if not bool(torch.isfinite(out).all()):
        raise UnsupportedRegional("Nonfinite blended x0")
    return out
