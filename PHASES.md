# Implementation plan — seven phases

## Goal and contracts
Build a **real Forge Neo extension** for native high-resolution txt2img using SDXL and Anima on constrained VRAM, with synchronized tiled denoising, global compositional coherence, a high-fidelity multilayer region editor, versioned CRF paste/import/export, regional prompts and LoRAs, and optional Forge-native structural guidance. The [design](docs/DESIGN.md) is the full technical/product contract; the [HTML prototype](docs/UI_PROTOTYPE.html) is the visual/interaction reference, **not functioning generation code**.

**One phase = one agent chat.** For each phase, the agent first drafts the detailed steps/checklist in `to-do.md` after reading the existing code, then implements, verifies and stops. Unfinished work stays in the same phase for the next chat. Do not skip acceptance items to move forward. Only promote a phase to DONE after its stated evidence exists; don't invent GPU test results.

**Agreed verification boundary:** P01–P06 use automated/offline evidence (unit, synthetic inference, code/API contracts, Chromium and isolated Gradio). **Only P07 performs live testing on the user's installed Forge Neo with real SDXL/Anima/Q4 hardware.** Missing live Forge access cannot block or reopen P01–P06. Never claim offline tests prove real model operation. Address discovered live-integration defects in P07.

This is the **simplified implementation grouping** of design §13's F00–F08 roadmap, not a scope reduction. The agent may refine internal sequencing where necessary without dropping requirements.

### P01 — Foundation, model probes, CRF, benchmark harness
**Depends:** none. **Initially READY.**
- Minimal loadable Forge Neo extension skeleton; inspect actual Forge Neo revision and extension hooks, SDXL/Anima sampling, positional metadata, cleanup, LoRA and native ControlNet APIs.
- Detect real Anima Q4 GGUF text-encoder path via existing loader; do not force unsupported replacement. Provide diagnostics and capability flags, including unsupported states.
- CRF v1 schema/semantic validator and box/polygon rasterization, sample imports, deterministic round-trip tests; preserve geometry, overlap and prompt inheritance explicitly.
- Build small CPU-only synchronized tile prediction tests and benchmark metadata recorder; minimal CI (syntax/schema/unit tests).
**DONE when:** extension imports via offline Forge stubs, model-probe contracts and CRF tests pass; real host discovery/model probes are reserved for P07. Do not claim GPU/image generation. Record the constraints for P02–P05.

### P02 — Production-quality regional editor and CRF workflow
**Depends:** P01.
- Implement the HTML reference as an integrated, responsive, non-destructive editor; preserve the visual design and interaction quality. All existing regions remain visible when drawing/editing others.
- Boxes, polygons, freehand paint/erase, selection, move/resize, overlays, layer order, visibility, locking, duplicate/delete, zoom/pan, undo/redo, inspector, per-region prompt/negative/adapter settings.
- Single canonical document state; autosave/project save/load; CRF JSON paste, validate, preview, merge/replace, export/import and meaningful errors, safe round-trips; asset masks if specified.
- Generation controls present real supported capabilities, and disclose inactive backend modes until integrated rather than pretending to generate.
**DONE when:** browser interaction tests cover 5+ persistent visible regions, repeated edits, undo, masks, CRF round-trip, reload and layout; a real isolated Gradio tab mount and browser interactions work; actual installed Forge page display is verified only in P07; no inert editor controls or silent state loss.

### P03 — SDXL synchronized low-VRAM tiled sampler
**Depends:** P01.
- Implement shared tile planner plus SDXL backend: one global noise/latent field, consistent sampler step/sigma across tiles, fixed per-step snapshot, weighted prediction accumulation, *one* global solver update.
- Safe CPU latent residency and sequential GPU work windows, global-coordinate tile mappings, overlap/halo handling, low-VRAM fallback, tiled VAE through Forge, interrupt/error cleanup.
- Test traversal-order independence, full coverage, seam behavior, determinism under controlled settings, memory telemetry; test SDXL interfaces with fake denoisers; defer real SDXL inference to P07.
**DONE when:** synchronized sampler, deterministic simulated comparisons, error/interruption handling and stubbed SDXL backend integration pass offline. Real SDXL image quality and VRAM remain unverified until P07.

### P04 — Global coherence and SDXL quality modes
**Depends:** P03.
- Implement a global-first/progressive mode (DemoFusion-inspired) and an efficient direct coherent tiled mode, reusing one scheduler; no independent finished-image stitching.
- Add explicit global scene/style distinction, spatial entity placement, patch-aware prompt routing (AccDiffusion-v2-inspired) and optional window/global interaction; allow manual/automatic/hybrid scene routing.
- Evaluate trade-offs and turn off any experimental algorithm that fails reproducible quality/memory tests; keep one robust working path.
- Add comparison scenes: one sun, lighthouse, crossing-tile bridge, two subjects, architecture, landscape, perspective, seams; test multiple seeds.
**DONE when:** offline routing, progressive/global-schedule and semantic-invariant tests pass, including simulated baselines. Actual image quality and VRAM improvements are evaluated in P07, not claimed here.

### P05 — Anima parity and Q4 encoder compatibility
**Depends:** P01, P03; integrate learnings from P04.
- Model-specific Anima DiT/flow backend with correct global positional handling (including RoPE), timestep/scheduler semantics, latent/VAE geometry and cleanup; do not transplant SDXL U-Net hooks.
- Preserve existing Q4 GGUF text-encoder integration; cache embeddings intelligently and offload only when supported; handle Anima variants via capabilities rather than model-name guesses.
- Support both high-res modes on Anima where technically validated; if an advanced mode needs a distinct algorithm, implement the compatible alternative and document it.
- Test Anima interface, timestep, positional and Q4 encoder-adapter contracts with source-backed stubs and synthetic tensors; real output, GGUF loader and VRAM measurements belong to P07.
**DONE when:** Anima/Q4 integration contracts and synthetic positional, scheduler and recovery tests pass offline. Mark real Anima/Q4 functionality unverified until P07, not an earlier-phase blocker.

### P06 — Full regional generation: prompts, LoRAs and controls
**Depends:** P02, P04, P05.
- Connect editor/CRF scene compiler to real per-tile/global conditioning for **both** SDXL and Anima. Handle overlaps, feathering, region crossing tile bounds and global defaults with deterministic semantics.
- Fast regional text attention; region-scoped LoRA behavior with capability-checked approximate and stronger prediction-blended isolation. Clearly distinguish LoRA leakage limitations. Treat text-encoder LoRAs distinctly from denoiser LoRAs.
- Optional Forge-native ControlNet/LLLite, including Anima-specific paths where available; shape/position correctness and limited-memory streaming.
- Test region/prompt routing, simulated overlap coefficients, multiple adapters, synthetic LoRA isolation, hook-conflict handling and rollback offline; real image-based duplicate/leakage assessments belong to P07.
**DONE when:** both model backends have offline-tested regional prompt/LoRA/control wiring, deterministic overlap handling and explicit capability failures. Actual image effects and leakage isolation remain unverified until P07; no silently ignored configurations.

### P07 — Full integration, packaging, acceptance and user handoff
**Depends:** P01–P06 accepted.
- Complete all remaining wiring: saved presets, generation metadata/reproduction, compatible Hires Fix/img2img behavior as documented (outside core native txt2img), resume if feasible, exception/OOM recovery, responsive UI and honest model/VRAM capability handling.
- **First and only live phase:** install in the actual Forge Neo environment; verify editor mount, SDXL/Anima/Q4 output, regional prompts/LoRAs, masks, controls, low-VRAM behavior, batching/interruption and extension interactions. Fix live issues here. Retain CI and document limitations.
- Produce a **self-contained installable extension ZIP** containing correct Forge extension folder and all dependencies needed for offline frontend operation. No stub or experimental-only package.
- Include `INSTALL_AND_TEST.md`: simple install by extracting into Forge Neo `extensions/` (or verified Extensions UI option), restart, choose SDXL/Anima, paste example CRF, run a small verified sample, then a high-res scene, inspect one-sun and region/LoRA tests; include rollback/removal.
- Publish tagged release/ZIP (or repository artifact) and exact verification results. Ask operator only for unavoidable live GPU/manual verification, not redundant exercises.
**DONE when:** the final ZIP and straightforward manual test instructions exist; all advertised core features work in validated configurations; any remaining unsupported optional modes are plainly identified. Do **not** declare release-ready if required Anima/SDXL paths are untested or failing.

## Simple workflow
- `main` holds completed accepted phases; use `phase/P0X-short-name` branches and one small PR per phase.
- Agent edits `to-do.md` to record detailed plan, status, tests and blockers. No external tracker required.
- CI: only quick Python/JS/schema/synthetic tests; **do not require online model weights/GPU** in GitHub Actions. Real Forge Neo/GPU/user-operated testing occurs **only in P07** and is recorded honestly.
- Avoid unnecessary new tooling, heavyweight process docs and risky replacement of Forge internals. Don't copy third-party source without checking license requirements.
- A future phase is NOT READY if a required predecessor has unresolved core gaps; resume the blocked predecessor instead.

## Primary sources
See `docs/DESIGN.md` §§2, 13 and 16 and `docs/research/`. Forge Neo, DemoFusion, AccDiffusion v2, Forge Couple, DiT coherence literature, and CRF are inputs, not license to claim tested behavior.
