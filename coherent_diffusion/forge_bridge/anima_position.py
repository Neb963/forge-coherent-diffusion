"""Source-shaped P05 Anima 3D RoPE with absolute patch coordinates.

Forge Neo backend.nn.anima.VideoRopePosition3DEmb forms [T*H*W,D/2,2,2]
and its DiT prepare_embedded_sequence starts every tile at 0. This bridge
substitutes only a guarded instance's embedded RoPE during a single forward.
No shared class/function is patched or kept changed after prediction.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch


class UnsupportedAnimaPosition(RuntimeError):
    pass


def _axis(position, frequencies):
    angle = position.float()[:, None] * frequencies[None, :]
    return torch.stack((torch.cos(angle), -torch.sin(angle),
                        torch.sin(angle), torch.cos(angle)), dim=-1)


def absolute_video_rope(embedder, *, t_tokens: int, h_tokens: int, w_tokens: int,
                        y_patch: int, x_patch: int, device):
    """Only allocates a *tile* RoPE; preserves native T/H/W token ordering.

    Compute local patch indices plus absolute read origin, not pixel positions.
    Frequencies follow the actual installed VideoRopePosition3DEmb instance,
    including its configured extrapolation ratios. No full-canvas GPU RoPE.
    """
    if any(type(n) is not int for n in (t_tokens, h_tokens, w_tokens, y_patch, x_patch)):
        raise UnsupportedAnimaPosition("DiT RoPE requires integral token positions")
    if min(t_tokens, h_tokens, w_tokens) < 1 or min(y_patch, x_patch) < 0:
        raise UnsupportedAnimaPosition("Invalid DiT RoPE token geometry")
    try:
        dh = embedder.dim_spatial_range
        dt = embedder.dim_temporal_range
        hf, wf, tf = (embedder.h_ntk_factor, embedder.w_ntk_factor,
                      embedder.t_ntk_factor)
    except AttributeError as exc:
        raise UnsupportedAnimaPosition("Unsupported Anima RoPE positional module") from exc
    if any(not isinstance(v, (int, float)) or not 0 < v < float("inf") for v in (hf, wf, tf)):
        raise UnsupportedAnimaPosition("Invalid native RoPE extrapolation")
    if not all(isinstance(v, torch.Tensor) and v.ndim == 1 and v.numel() > 0
               for v in (dh, dt)):
        raise UnsupportedAnimaPosition("Unexpected native RoPE dimension vectors")
    # Same float16 frequency construction as Forge Neo's source implementation.
    hfreq = 1.0 / ((10000.0 * hf) ** dh.to(device=device, dtype=torch.float16))
    wfreq = 1.0 / ((10000.0 * wf) ** dh.to(device=device, dtype=torch.float16))
    tfreq = 1.0 / ((10000.0 * tf) ** dt.to(device=device, dtype=torch.float16))
    t = _axis(torch.arange(t_tokens, device=device), tfreq)
    h = _axis(torch.arange(y_patch, y_patch + h_tokens, device=device), hfreq)
    w = _axis(torch.arange(x_patch, x_patch + w_tokens, device=device), wfreq)
    # Native cat order: temporal, height, width; C-order flatten T,H,W.
    tempor = t[:, None, None].expand(-1, h_tokens, w_tokens, -1, -1)
    height = h[None, :, None].expand(t_tokens, -1, w_tokens, -1, -1)
    width = w[None, None, :].expand(t_tokens, h_tokens, -1, -1, -1)
    out = torch.cat((tempor, height, width), dim=-2)
    return out.reshape(t_tokens * h_tokens * w_tokens, -1, 2, 2).float()


@contextmanager
def scoped_absolute_rope(core, tile, *, patch_spatial: int):
    """Temporarily replace only this exact native DiT instance's position return.

    The instance must have an *unmodified* native prepare method. No reentrant
    access and no concurrently-owned generations. External Forge generation
    using the same DiT instance concurrently remains unsafe until P07 host gate;
    a caller MUST hold exclusive model ownership around this scope.
    """
    if (type(core).__name__ != "Anima"
            or type(core).__module__ != "backend.nn.anima"
            or getattr(core, "patch_spatial", None) != patch_spatial
            or getattr(core, "patch_temporal", None) != 1):
        raise UnsupportedAnimaPosition("Unsupported Anima DiT model/patch layout")
    original = getattr(core, "prepare_embedded_sequence", None)
    if not callable(original) or "prepare_embedded_sequence" in vars(core):
        raise UnsupportedAnimaPosition("DiT position method already patched or unavailable")
    if getattr(core, "_coherent_rope_active", False):
        raise UnsupportedAnimaPosition("Concurrent/reentrant Anima RoPE context")
    if tile.read.x0 % patch_spatial or tile.read.y0 % patch_spatial:
        raise UnsupportedAnimaPosition("Global read origin is not patch-aligned")
    setattr(core, "_coherent_rope_active", True)

    def adjusted(self, latent, padding_mask=None):
        embedded, local_rope, extra = original(latent, padding_mask=padding_mask)
        if embedded.ndim != 5 or embedded.shape[1] != 1:
            raise UnsupportedAnimaPosition("Only single-frame Anima RoPE supported")
        t, h, w = (int(embedded.shape[1]), int(embedded.shape[2]),
                   int(embedded.shape[3]))
        if (tuple(latent.shape[-2:]) != (tile.read.height, tile.read.width)
                or h * patch_spatial != tile.read.height
                or w * patch_spatial != tile.read.width):
            raise UnsupportedAnimaPosition("DiT embedded token window disagrees with tile geometry")
        rope = absolute_video_rope(
            self.pos_embedder, t_tokens=t, h_tokens=h, w_tokens=w,
            y_patch=tile.read.y0 // patch_spatial,
            x_patch=tile.read.x0 // patch_spatial,
            device=latent.device)
        if tuple(rope.shape) != tuple(local_rope.shape):
            raise UnsupportedAnimaPosition("Native DiT RoPE tensor layout mismatch")
        return embedded, rope, extra

    try:
        core.prepare_embedded_sequence = MethodType(adjusted, core)
        yield
    finally:
        if "prepare_embedded_sequence" in vars(core):
            del core.prepare_embedded_sequence
        if hasattr(core, "_coherent_rope_active"):
            del core._coherent_rope_active
