# P03 — Offline SDXL synchronized tiled sampler

**Evidence tier:** PyTorch CPU synthetic contract + isolated Forge Neo source-shaped adapter fakes only. This is **not** a real SDXL model quality/VRAM result, not a user-facing Forge generate path, and not Anima support. Installed Forge Neo, GPU, real checkpoint inference, text-encoder operation and actual tiled VAE decode are tested only in P07.

## What this phase adds

- `coherent_diffusion.scheduler.sdxl_sync.TilePlan`: image-space tile/overlap/halo values, explicit 8× SDXL latent mapping, clipped context reads and write cores. The planner rejects misaligned dimensions, negative/invalid overlaps, excessive windows and missing/duplicate geometry.
- `seeded_noise`: one CPU float32 4-channel latent noise field, one RNG seed for the entire image, never regenerated separately for tiles.
- `synchronized_euler`: a full-canvas frozen latent for each sigma interval; sequential window inference; clipped read halo; write-core-only positive cosine weights; CPU float32 numerator/denominator; exactly one global Euler update and one global sigma schedule. A window predictor receives `(crop, sigma, Tile)` and must return **denoised x0 on the same read-crop shape**, not epsilon or noise.
- `forge_bridge.sdxl.ForgeSDXL`: source-backed Forge API contract for base `StableDiffusionXL` only. It invokes the installed model's **existing** `get_learned_conditioning`, Forge `compile_conditions`, `sampling_prepare`, `sampling_function_inner` and `sampling_cleanup`. Direct tiled VAE decode uses the host VAE's own `process_out` followed by `decode_tiled`; there is no custom VAE or CLIP loader.
- Explicit model/capability refusal: Anima, SDXL refiner, inpaint/concat, SDXL rectified-flow variants, extra model hooks and active ControlNet/LLLite must not silently pass. This path owns **Euler ODE for canonical denoised x0** only. Ancestral, DPM++, special CFG/model hooks and accelerated profiles require separate, model-specific proof and are not quietly replaced with Euler.
- Memory policy: full latent, weighted accumulator and denominator on CPU; one halo crop on the requested inference device. Preflight rejects declared global CPU budget before noise/model allocation. A failed inference tile may trigger **up to two bounded complete-run retries** with a smaller tile/overlap/halo, the same seed and a fresh global solver trajectory. The actual effective geometry is in `RunResult.plan`; estimates are labeled as estimates, never measured VRAM.

This does **not** make independent txt2img requests for each crop. It also does not prove that synchronized local predictions alone solve duplicates, scene consistency, horizon/perspective artifacts or seams in generated images. P04 implements additional global coherence and evaluation.

## Programmatic API (future P04/P07 caller)

```python
from coherent_diffusion.scheduler import TilePlan
from coherent_diffusion.forge_bridge.sdxl import ForgeSDXL

# The host must already own a loaded, supported Forge StableDiffusionXL model.
plan = TilePlan.build(
    width_px=2048, height_px=1536, tile_px=768,
    overlap_px=128, halo_px=64,
)
backend = ForgeSDXL.from_prompts(
    host=loaded_forge_sdxl, plan=plan,
    prompt="landscape", negative_prompt="blur",
    cfg=5.0, seed=123456,
)
# Supply an actual validated host-compatible sigma schedule for Euler ODE;
# this illustrative list does NOT claim any recommended model schedule.
run = backend.run(sigmas=[1.0, 0.5, 0.0])
print(run.plan.tile_px, len(run.steps), run.cpu_workspace_estimate_bytes)
pixels_nhwc = backend.decode_tiled(run.latent)
```

This is an **illustration of the API**, not proof that a particular sigma sequence is appropriate for a real SDXL checkpoint. A production UI must obtain and validate its actual schedule/model-specific parameterization, queue ownership and reproducibility metadata before exposing a Generate action. This P03 module deliberately does not monkeypatch Forge's generation/sampler. All existing P02 editor controls remain editor-only until P06/P07.

## Verified and unverified details

- Contract reference: Forge Neo `Haoming02/sd-webui-forge-classic` `neo` source (reviewed 2026-10-25): `backend/diffusion_engine/sdxl.py` supplies SDXL conditioning `crossattn` and `vector`; `backend/sampling/condition.py` compiles `c_crossattn`/`y`; `backend/sampling/sampling_function.py` implements `sampling_function_inner` and `sampling_prepare/cleanup`; `backend/patcher/vae.py` exposes `decode_tiled`.
- Offline tests: `python -m unittest discover -s tests -p 'test_sdxl_sync.py' -v` with torch CPU, plus pre-existing Python schema/CRF tests, Chromium editor and isolated Gradio tests.
- Coverage includes seeded determinism, geometry and halo, overlap numerator/denominator, frozen state, shuffled window invariance, multi-step Euler reference, malformed sigma/nonfinite tensors, CPU/window budget refusal, interruption, mid-prediction error and cleanup, bounded simulated OOM retile with replay from the original seed, and fake Forge model API call ordering.
- No actual VRAM allocation/peak data are collected. `RunResult.cpu_workspace_estimate_bytes` and `max_window_latent_bytes` are geometric budget estimates, not runtime memory measurements. Forge's activation memory and VAE transient allocations are not captured; success on 4 GB VRAM is not established.
- **Remaining phases:** P04 progressive/direct semantic global correction; P05 Anima-specific flow/RoPE/Q4 path; P06 region/LoRA/control routing; P07 actual integrated Forge generate UI, installed runtime and real-image/hardware acceptance and final extension ZIP.
