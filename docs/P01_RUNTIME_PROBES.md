# P01 — Source-grounded Forge Neo probes and runtime evidence

## Source inspected, not the installed application
Upstream: Haoming02/sd-webui-forge-classic, neo commit
0c388f8b26d26d475c2c7104aea4eafdf7bbb3e7 (2026-10-23).

- modules/scripts.py: extension-basedir imports; Script.title/ui/show/process,
  AlwaysVisible and stock generation callback API exist.
- modules/script_callbacks.py: CFG denoiser, denoised, after-CFG hooks are present.
  P01 installs none and makes no sampler or model patches.
- backend/diffusion_engine/base.py: forge_objects, model_config,
  get_learned_conditioning, and VAE encoding/decoding.
- backend/diffusion_engine/anima.py: Anima Qwen06Engine / CLIP, VAE, UnetPatcher;
  backend/text_processing/anima_engine.py uses the host's own text encoder.
- backend/loader_gguf.py: a GGUF loader is present. This does not prove that
  Anima Q4 GGUF is in use or that a real encode succeeds.
- Positional/RoPE remapping, solver prediction field, per-step sigma timeline,
  LoRA and ControlNet compatibility are unverified.

## Live integration check — only when user has a Forge host
1. Clone repo directly into Forge Neo extensions/coherent-diffusion; restart.
2. In txt2img expand "Coherent Diffusion — diagnostics only". If missing, report
   installed Forge revision and traceback. CI cannot prove successful host load.
3. Select an SDXL or Anima model normally, then press "Inspect loaded model".
   This reads existing object/class metadata only, without loading weights.
4. Optionally enable "Capture read-only metadata during txt2img", run a *normal*
   Forge generation and click Inspect again. Request/sampler metadata comes
   from Script.process. No prompt text, patch, hook, or tensor data is captured.
5. Q4 GGUF: inspect the user's already selected text encoder, then separately
   run a normal Anima encode. An observed gguf_cls parameter is only indirect
   loader evidence. P01 does not assert Q4 or successful encode.
6. Compare stock Forge disabled vs enabled behavior. The P01 panel does not
   implement image generation, synchronized sampling, or regional editor UI.

## CRF v1 conventions
- Normalized [0,1] top-left-origin full-canvas coordinates; bbox is
  [x,y,width,height], not two corners.
- Rasterizer samples pixel centers. Bbox upper bounds exclusive, polygon uses
  even-odd rule with boundary included. Reject degenerate/self-crossing shapes.
- Prompt missing = unspecified, null = deliberately unset, empty string =
  explicitly blank. The validator and serializer preserve distinctions and
  never insert full global scene text into local prompts.
- Enabled controls generation intent; visible is UI only; priority uses
  z_index and then original document order.
- Base weight zero requires a positive enabled full-canvas bbox: conservative
  exact coverage, not an approximate pixel-grid guess. Some complete unions
  may be rejected until a future exact-coverage compiler.
- Feather, LoRA isolation and region-weight normalization are NOT implemented
  by P01; future backend must enforce them.
- Binary mask manifest entries are metadata-only, unverified until P02 ZIP
  decoding, path/hash/size checks; such assets must never reach a sampler
  without validation. No external URL fetching or code execution.
- Reject future version and unknown fields outside inert extensions namespace.

## Remaining proof obligations
P02: browser editor, ZIP asset verification, undo/reload/aspect previews.
P03: real SDXL sampler, frozen global tensor, one update/step, memory policy.
P05: Anima 3D RoPE/sigma/encoder path real model proof including Q4 GGUF.
P06: numeric region blend, LoRA/control integration and cleanup.
All phases: no assertion of tested runtime behavior from offline mocks alone.

Evidence tiers at authoring time: Tier 1 tests pending CI; Tier 2 real Forge
smoke NOT TESTED; Tier 3 real GPU generation/VRAM NOT TESTED.
