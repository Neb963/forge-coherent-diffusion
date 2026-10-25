# P04 — SDXL global-first quality and semantic routing

**Evidence tier:** offline/synthetic, **not** installed Forge Neo, checkpoint, GPU, or perceptual image-quality validation. P07 is the sole live-test phase. P03's direct synchronized SDXL core is preserved unchanged. This code is opt-in and does not register a generation UI.

## Implemented high-resolution execution policies

| Policy | Trajectory | Available now | Constraint |
| --- | --- | --- | --- |
| `direct` | One seeded target-resolution latent; every P03 window reads a frozen global step and all predictions receive one full-canvas Euler update | Opt-in Python API | Coherent update, **not guaranteed coherent scene semantics** |
| `progressive` | One low-resolution **single-window global denoising** pass, bilinear latent lift, seeded full-target noise at the refinement initial sigma, then P03's synchronized generative target-resolution diffusion | Opt-in Python API; stage-specific SDXL bridge | Requires bounded, aligned anchor size and two explicit schedules; extra inference, not pure target-resolution-from-noise |
| Experimental sparse/dilated or FFT-style interaction | No executable model path | **Disabled** | Requires image-based quality/VRAM evaluation before enabling; the contract rejects non-`none` values |

Global-first changes the initial conditions for the target denoising pass; it is not independent tile rendering or an ESRGAN-only upscale. The anchor may carry global composition, but there is **no guarantee of one sun, correct anatomy, or continuous structures** on real SDXL output.

`QualityPlan.build(target, mode="progressive", refine_sigmas=..., anchor_sigmas=..., anchor_factor=2)` creates two explicit geometries. Anchor factors are 2 or 4, with exact latent alignment and a 1,048,576-image-pixel anchor budget by default; the anchor must fit as **one complete global denoiser window** under its separately checked memory budget. A larger anchor is rejected rather than silently tiled and misrepresented as global. Both schedules must strictly descend to sigma 0. Different independent sigma trajectories are deliberate; these are explicit algorithm parameters, not a Forge sampler preset.

`execute_quality(quality, seed, predict, ...)` expects a stage-labelled **denoised-x0 predictor**: `predict(crop, sigma, tile, stage)`, with `stage` one of `anchor`, `refine`, `direct`. One CPU RNG seed controls the anchor (or direct target); a separately derived deterministic seed controls the global high-resolution refinement noise. Neither stage changes the PyTorch global RNG. Stage progress includes full geometry, sigma-step records, predicted forward-call counts and **estimated** CPU/window tensor footprints, not measured RSS/VRAM. A single outer cleanup executes on interruption, preflight, bad model output or other exception. The P03 tensor scheduler remains the sole per-step solver.

`run_forge_sdxl_quality(host, quality, prompt=..., negative_prompt=..., cfg=..., seed=..., device=...)` in `coherent_diffusion/forge_bridge/quality.py` is an opt-in wrapper around the original P03 `ForgeSDXL`. It uses *stage-specific* prompt size conditioning, same loaded host, and prepare/cleanup on each stage transition; it does not alter loaders, register hooks or silently switch samplers. A failed second stage produces no successful result. P04 does **not** implement adaptive retile replay for progressive mode; P03's standalone direct adapter retains that option. The SDXL bridge's actual denoiser/sigma compatibility and VRAM handling require P07 verification.

## Scene-aware window routing plan

`compile_routes(crf_doc, target_tile_plan, mode="manual" | "automatic" | "hybrid")` validates CRF v1 before compiling deterministic **advisory** routes.

- The full `global.scene_prompt` is retained for anchor/global context, while reusable `global.style_prompt` and applicable local region prompts form each window's proposed text. The full global scene is **never repeated verbatim per window**.
- For authored boxes and polygons, intersection is computed against each window's global-coordinate **read area** (including halo). Polygon intersection uses area-positive rectangular clipping; a mere bounding-box false positive does not route an entity. Disabled or zero-weight regions are excluded. Editor `visible` does not affect generation intent.
- Entities keep their **single CRF ID and count intent** across every intersecting window; an object crossing several tiles is not rewritten into several independently numbered copies. Their reported union bounding box is authored geometry, not a detected object silhouette.
- `manual` routes authored regions/entities. `automatic` associates simple count expressions with **uniquely name-matched authored region IDs/names only**, preserving existing entity IDs/counts and warning on conflicts; unmatched/ambiguous nouns remain **unlocated** proposals (confidence 0.25). `hybrid` keeps all manual routes plus those conservative matches. Automatic mode never invents object positions or uses unverified masks; it is not learned scene detection.
- Unsupported binary mask asset routing is refused rather than reduced to a bounding rectangle. Actual per-tile text encoder attention, masked LoRAs, ControlNet/LLLite, model-specific region encoding and error handling are P06 work. `connected_to_model=False` is explicit and must not be presented as an active generation feature.

## Test scope and interpretation

The dedicated CPU PyTorch job (pinned torch 2.5.1+cpu, validator from `requirements.txt`) executes `test_quality_p04.py` and `test_forge_quality_p04.py`, in addition to P03's original suite. Coverage includes:

1. Frozen synchronized target steps, seeded reproducibility, stage/forward trace, global one-window anchor, deterministic noise lift/refinement, schedules and memory preflight.
2. Model failures, nonfinite tensors, cancellation and exact cleanup, stage-correct Forge prompt sizes and sequential fake prepare/cleanup.
3. Authored box/polygon intersection, split-window entity identity/count continuity, negative-prompt states, disabled versus visible regions, manual/auto/hybrid placement, ambiguity/conflicting counts, abstention and missing geometry errors.
4. Eight **synthetic topology scenes** (one sun, lighthouse, bridge spanning tiles, two subjects, architecture, landscape, perspective, seams), three seeds each. A separate toy Gaussian predictor deliberately duplicates objects when the whole scene is independently sent to every window. The global-anchor simulation retains a single blob. This tests algorithmic wiring **only**: it does **not** establish that a trained diffusion network behaves this way or yields higher-quality images.

No actual photos, GPU peak measurements, live Anima paths, receptive-field calibration, true learned global interactions or real SDXL tensor outputs are evaluated in P04. P05 implements distinct Anima positional/flow semantics; P06 attaches semantic routes to actual model conditioning; P07 evaluates output quality, performance, VRAM, failure recovery and all installed Forge integrations.

## API example (synthetic only)

```python
import torch
from coherent_diffusion.scheduler.sdxl_sync import TilePlan
from coherent_diffusion.scheduler.quality import QualityPlan, execute_quality

tiles = TilePlan.build(1024, 768, 512, overlap_px=128, halo_px=64)
quality = QualityPlan.build(
    tiles, mode="progressive",
    anchor_sigmas=(1.0, 0.5, 0.0),
    refine_sigmas=(0.3, 0.15, 0.0),
    anchor_factor=2,
)
# For numerical demonstration only. A real backend must return denoised x0.
result = execute_quality(quality, seed=42,
                         predict=lambda crop, sigma, tile, stage: crop * 0.5)
assert result.latent.device.type == "cpu"
```

Do not mistake this identity-like toy predictor for a model generation example.
