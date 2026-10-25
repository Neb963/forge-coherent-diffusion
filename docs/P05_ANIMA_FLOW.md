# P05 — Anima flow, global 3D RoPE, and existing Q4 encoder route

**Evidence tier: offline PyTorch 2.5.1+cpu, synthetic DiT tensors and source-shaped Forge Neo fakes.** Real Forge Neo, actual Anima model, user-installed GGUF loader/Q4 encoder, device VRAM/performance and generated image fidelity **NOT TESTED until P07**. These APIs are opt-in only; the editor **does not Generate** yet.

## Architecture and source contract

The P03/P04 SDXL implementation remains unchanged. Forge Neo's \`backend.diffusion_engine.anima.Anima\` owns a WAN VAE, \`backend.nn.anima.Anima\` DiT, \`KModel\` constant-flow predictor and existing \`Qwen06Engine\`. This is **not** a four-channel 2D U-Net. The Anima path uses CPU-resident \`[B,16,1,H,W]\` (one-frame) latents with an 8x spatial WAN VAE scale and native patch-aligned read windows.

- \`scheduler/anima_flow.py\`: \`flow_sigmas\` validates **explicit** strictly falling [0,1] sigma schedules; never rewrites host model-shift settings. \`anima_noise\` generates one seeded 5D full-scene latent. \`synchronized_flow\` reads a frozen global state for all overlapping windows at each sigma, combines model-derived denoised x0 with a positive cosine weighting, then makes **one global Euler-flow update** \`x_next=x+(sigma_next-sigma)*(x-x0)/sigma\`. Same global field/seed/time for all windows; no completed-tile image assembly. Uses windowed device transfer, CPU float32 accumulation, strict 16-channel/patch/memory preflight and scoped failure/cancellation cleanup.
- \`AnimaQualityPlan\` and \`execute_anima\`: explicit **direct** and **progressive** modes, each with its own validated flow schedule. Progressive runs a single **whole-scene DiT anchor** at 1/2 or 1/4 target image dimensions, lifts its latent using spatial bilinear interpolation, adds one separately seeded full-target noise field, then executes generative whole-image synchronized refinement. Memory/handoff checks run before the first stage. The anchor requires a single model window; no silently substituted tiling. Experimental global interaction modes are refused.
- \`forge_bridge/anima_position.py\`: native \`VideoRopePosition3DEmb\` computes **tile-local** T/H/W indices. \`absolute_video_rope\` reconstructs the native frequency/axis layout using absolute **patch** coordinate offsets from the tile's read origin, including its halo. It allocates RoPE only for that window, not the full GPU canvas. \`scoped_absolute_rope\` injects this into exactly one **native Anima instance** during a single synchronous prediction; restores the original method even on error and forbids reentrant/externally patched interfaces. Requires exclusive host model ownership and unchanged source-shaped DiT positional implementation; concurrency and other model variants remain P07/P05 fail-closed gates. This is not a global Forge monkey patch.
- \`forge_bridge/anima.py\`: \`ForgeAnima.from_prompts\` calls the **existing host** \`get_learned_conditioning\`, unpacks Anima's list of tensor embeddings and runs Forge's \`compile_conditions\`. The exact same installed CLIP/Qwen/Q4 text-loader instance remains owned by Forge. An instance-scoped cache holds positive and negative conditioning only for its current direct/progressive run. No disk/GGUF file assumptions, no encoder dequantization, no alternative loader, no unilateral offloading. \`sampling_prepare\`, \`sampling_function_inner\` and \`sampling_cleanup\` are the Forge entrypoints; \`KModel\` converts velocity into denoised x0. \`decode_tiled\` uses the host WAN \`first_stage_model.process_out\` then 3D \`vae.decode_tiled\` and returns RGB NHWC images by removing the single time frame.

## Fail-closed capabilities

P05 accepts **native source-shaped** Anima, one-frame temporal patch = 1, 16-channel WAN VAE, 8x XY scale, a known constant-flow predictor and native 3D RoPE. It rejects other architectures, img2img/reference latents, inpainting, extra concatenation, ControlNet, unnegotiated model hooks and incompatible/misaligned DiT patch geometry. Multi-frame/video Anima, custom/new RoPE implementations, model-level cross-window attention, learned global quality corrections and exotic samplers are **not** claimed to work. P06 owns regions/LoRAs/controls and conflict negotiation.

Native \`sampling_prepare\` controls model residency. The CPU-window budget covers **latent data only**, not actual device activations/model weights. CPU memory estimates are estimates. There are no claimed GPU measurements, no automatic precision override and no replacement of the Q4 GGUF loader. Host reference-image state may be global: even though the scoped position method is restored, P07 must validate exclusive generation ownership and the actual reference/extension lifecycle. Predictors must return valid **denoised x0** at the exact signed sigma, not raw flow velocity.

## Synthetic API example

\`\`\`python
from coherent_diffusion.scheduler.sdxl_sync import TilePlan
from coherent_diffusion.scheduler.anima_flow import AnimaQualityPlan
from coherent_diffusion.forge_bridge.anima import run_forge_anima_quality

tiles = TilePlan.build(1024, 768, 512, overlap_px=128, halo_px=64)
quality = AnimaQualityPlan.build(
    tiles, patch_spatial=2, mode="progressive", factor=2,
    anchor_sigmas=(1.0, 0.5, 0.0),
    refine_sigmas=(0.35, 0.175, 0.0),
)
# In P07, host must be the *already loaded* supported Forge Anima model.
# P05 never activates a Generate control or loads host/model weights.
# result = run_forge_anima_quality(host, quality, prompt="landscape", seed=123)
\`\`\`

Patch size 2 is illustrative, **not** a claim about every Anima checkpoint. Query the loaded DiT's actual \`patch_spatial\` and build aligned tiles. The example sigma trajectory is a synthetic numerical schedule; real checkpoint-appropriate shifted schedules and visual results require P07.

## Offline tests

In the pinned CPU-torch GitHub Actions job: \`python -m unittest discover -s tests -p 'test_anima_p05.py' -v\`. Tests cover five-dimensional one-frame memory/shape, sigma rejection, numeric single-step/frozen-step reference, shuffling and overlapping windows, original-input preservation, OOM/cancel/nonfinite cleanup, reproducible two-stage anchor/refine, full-grid versus absolute sub-grid positional agreement, exact restoration on exceptions, native WAN decoder layout, existing Q4-like encoder identity/no replacement, fake Forge stage prepare/predict/cleanup and capability rejections. Existing SDXL/CRF/Chromium/Gradio regressions remain enabled.

**P06:** connect both backends and CRF scene compiler to actual per-region conditioning, LoRA separation and optional controls. **P07:** real installed Forge Neo, SDXL/Anima Q4, image output, user-device VRAM, true RoPE/flow test and extension conflicts.
