# Coherent Diffusion — P07 candidate install and live acceptance

**Status: integration candidate, NOT release-ready.** The ZIP includes Python modules and the locally rendered HTML editor, but actual Forge Neo SDXL / Anima Q4 / GPU acceptance has **not** been performed by this agent. P07 remains open pending these results. Do not replace an accepted stable extension with an untested candidate without taking a backup.

## Installation and rollback

1. Confirm Forge Neo is stopped. Back up any existing `extensions/coherent-diffusion` directory. Your saved editor drafts live in the browser profile, not inside the extension directory; export a CRF ZIP before changing browsers.
2. Extract `coherent-diffusion-P07-candidate.zip` **into** Forge Neo's `extensions/` directory. The resulting path should be `extensions/coherent-diffusion/scripts/coherent_diffusion.py`; do not nest the directory twice.
3. In the exact Python environment that launches Forge, ensure `jsonschema`, `torch`, `Pillow` and Gradio are installed. The repo's `requirements.txt` pins jsonschema, but the extension intentionally reuses Forge's existing Torch/Gradio model environment. Do not replace or reinstall your working GGUF/Q4 loader. There are no CDN/frontend dependencies.
4. Restart Forge Neo. Open **Coherent Diffusion**. The editor remains a full-width tab; generation results/errors appear in the **Generate · loaded Forge checkpoint** panel under it. If the tab fails to load, record the Forge log.
5. **Rollback:** stop Forge, remove `extensions/coherent-diffusion`, restore the original backup if any, and restart. No persistent Forge checkpoint/loader patch is intended. Retain exported CRF ZIPs.

## Minimal live acceptance (operator on the actual installed host)

Keep existing SDXL / Anima / Q4 and native Forge installations unchanged.

- **Host integration:** With the extension enabled, confirm the canvas mounts, five regions remain visible after editing a sixth, CRF JSON/ZIP mask import/export survives refresh, and clicking the canvas **Generate image** transfers composition/settings to the native Gradio Generate panel. Confirm a normal txt2img generation works with the extension enabled but unused.
- **SDXL direct & progressive:** Select an already-loaded supported SDXL checkpoint, start with one 512×512 or 1024×1024 region-free case, then use a 2048 scene. Confirm actual output, sampler schedule quality, coherent scene identity, crop/seam continuity and peak VRAM/RAM using Forge logs. The schedule is derived from the loaded SDXL predictor's discrete training sigma table, but its actual generation behavior and image quality must be validated on the installed checkpoint.
- **Anima/Q4 direct & progressive:** Keep the existing Q4 GGUF text-encoder loader untouched. With an already-loaded Anima checkpoint, generate a small native-resolution case then a tiled case; verify successful *actual encoder invocation*, temporal shape, global RoPE positioning, WAN tiled decode and no Q4 reload/conversion. Compare against native Forge output and inspect failures.
- **Regions:** Import `examples/one_sun.crf.json` after adapting its canvas to the chosen output dimensions where necessary. Generate overlapping / crossing-tile regions, PNG mask ZIP, alternate negatives and two seeded scenes; inspect if promised region effects are real, not just source-shaped contracts.
- **Isolation/controls:** Regional LoRA evaluation is **currently gated OFF** until whole-Forge exclusive ownership is established. Built-in native ControlNet / LLLite / VACE control factories are **not yet wired**; unsupported modes must fail explicitly. Implement and retest these before P07 can pass its full product contract. Do not treat these as successful merely because synthetic tests passed.
- **Recovery and coexistence:** Check cancel during a long tiled forward, request a second run, force a reproducible memory failure, verify no stale hooks/memory leak after repeated jobs, and run plain Forge txt2img again. Verify Hires Fix/img2img remain native-only (no silent compatibility claim), check LoRA Control/Forge Couple conflicts, and capture actual extension output metadata.
- **Release gate:** Record installed Forge revision, model/encoder source and hashes, Q4 route, GPU and memory configuration, seed, scene, resolution, peaks/elapsed time, images, observed defects and patched retest evidence. Only publish a stable release when both real SDXL and Anima/Q4 work and required P07 controls/LoRAs/cleanup are verified. Do **not** mark DONE based on this candidate ZIP or CI alone.

## Current constraints

Generation is opt-in, batch size 1, exact output sizes divisible by 64 up to 4096, 256–1024 px 64-aligned tile sizes, progressive anchor factor 2. A validated CRF ZIP is required for painted masks. Automatic model downloads, arbitrary third-party prompt hooks, source Hires Fix/img2img processing, checkpoint resume, regional text-encoder LoRAs, native external control model loading and unverified non-Euler schedulers are not supported. External shared LoRA patch ownership remains unverified. The parent-panel image is a result preview; the source document is not automatically embedded in the PNG. Hash/seed/geometry/forward estimates are reported in the JSON evidence area, not measured GPU performance. A user must independently export the original CRF ZIP for replay.

Python offline checks and Gradio/Chromium tests are **not** proof that the user's installed Forge/Q4/GPU paths work.
