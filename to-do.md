# TODO — phase tracker

Read `PHASES.md` first. **Exactly one phase per chat.** Mark DONE with phase-appropriate evidence: **P01–P06 offline only, P07 live Forge Neo/model acceptance only**. Missing live access cannot block earlier phases.

| Phase | Status | Depends on |
|---|---|---|
| P01 — Foundation, probes, CRF, tests | **DONE (offline acceptance; live Forge deferred to P07)** | — |
| P02 — Regional editor and CRF UI | **DONE — offline acceptance 2026-10-25; live Forge deferred to P07** | P01 |
| P03 — SDXL synchronized tiled generation | **DONE — offline acceptance 2026-10-25 (live P07)** | P01 |
| P04 — SDXL global coherence | READY | P03 |
| P05 — Anima + Q4 GGUF parity | BLOCKED | P01, P03, P04 findings |
| P06 — Regional prompts, LoRAs, controls | BLOCKED | P02, P04, P05 |
| P07 — Final package and simple manual guide | BLOCKED | P01–P06 |

## Active phase — P01: Foundation, model probes, CRF, benchmark harness
**Status:** DONE after offline tests; matching Forge runtime unavailable. Branch `phase/P01-foundation`.

### Pre-implementation inspection (2026-10-24)
- Main contains specs and an HTML reference only; no extension, dependencies or tests.
- Forge Neo source inspected at `Haoming02/sd-webui-forge-classic` branch `neo`, upstream commit `0c388f8b26d26d475c2c7104aea4eafdf7bbb3e7` (2026-10-23). Relevant host interfaces: `modules/scripts.py` Script/UI/process/postprocess, `modules/script_callbacks.py` callbacks, `backend/diffusion_engine/anima.py`, `backend/text_processing/anima_engine.py`, `backend/loader_gguf.py`. This is **source inspection only**, not a check of the user's installed Forge or GGUF loader. Never replace encoder path.
- No Forge installation or GPU/model runtime is exposed in this execution environment: record actual runtime probing as NOT TESTED. Offline validation must still run.

### Concrete implementation plan — written before code
- [x] Create a minimal loadable Forge `scripts/` Script with a read-only diagnostics panel, graceful no-model/error states, scoped lifecycle observation (no persistent monkey patches, no generation override).
- [x] Add host probes that report model/encoder object types and qualified paths, `forge_objects` patchers, conditioning entry-point, sampler/callback/position support as supported/unknown; explicitly report inability to **prove** Q4 GGUF from a file extension or object name alone. Include compatible runtime source revision and cleanup evidence.
- [x] Define portable CRF v1 Draft 2020-12 schema for exact example format; conservative bounds and extensible namespace; strict JSON/fenced parsing with duplicate-key/NaN rejection, geometric/ID/overlap/asset semantic checks and explicit missing/empty/null prompt handling; no external execution/download.
- [x] Deterministic CPU-only bbox/polygon masks with documented edge/pixel convention, self-intersection rejection, positive coverage; canonical JSON round-trips and at least 10 valid/20 invalid conformance fixture documents.
- [x] Implement small synchronized tile prediction reference harness: frozen global snapshot, positive-coverage weighted accumulator, one global update and order-invariance tests (explicitly not a production sampler).
- [x] Add bounded benchmark metadata recorder (no fake timing/memory results), rapid unittest/schema/compile checks in GitHub Actions, short run/test/probe documentation.
- [x] Execute deterministic tests and inspect GitHub Actions at the exact commit. Run real Forge smoke only if a matching host is actually accessible; report it NOT TESTED otherwise.
- [x] Update status/evidence in `to-do.md`, publish code on P01 PR; squash-merge only if acceptance is satisfied. Stop without beginning P02.

### Work log / evidence (2026-10-24)
- Original source-only repo inspected; phase plan committed **before** implementation in commit `859308114bed261ec5dde390433b4c148caffc00`.
- Implemented Forge `scripts/coherent_diffusion.py` P01 read-only diagnostics; host model detection via `sd_models.model_data.sd_model` or opt-in `Script.process(p)`. No sampler override/model patches/callback registration. CI mocks prove script API shape, **not that the UI loads in the actual Forge installation**.
- CRF Draft 2020-12 schema, strict duplicate-key/NaN/fence parser, exact semantic guards for bboxes/polygons/IDs/references/assets/base coverage, non-destructive canonical JSON and 11 valid / 28 invalid conformance documents. Binary mask manifest allowed but hash verification and raster import deferred to P02, explicitly warned.
- Deterministic CPU pixel-center box/polygon rasterizer, overlap isolation, synchronized frozen-snapshot CPU tile prediction accumulator with traversal invariance and full coverage checks. This is NOT a Forge sampler.
- Bounded benchmark metadata with explicit `not_tested` status. Wrote honest upstream source/API reconnaissance to `docs/P01_RUNTIME_PROBES.md`; inspected Forge Neo commit `0c388f8b26d26d475c2c7104aea4eafdf7bbb3e7` (not user's runtime).
- GitHub Actions: [branch test run 40127871046](https://github.com/Neb963/forge-coherent-diffusion/actions/runs/40127871046), commit `f44dad3198d86a39cdefa440a5180d99140b0a91`, completed **SUCCESS** on Ubuntu Python 3.11. Commands `python -m compileall -q coherent_diffusion scripts tests` and `python -m unittest discover -s tests -v`; **19 tests PASS**. CI validates schema and all 39 fixture documents.
- No Forge application/runtime, GPU, SDXL checkpoint, Anima checkpoint, or installed Q4 GGUF text encoder was accessible from this execution environment. Actual extension import, encoder inference, positional context, sampler lifecycle and VRAM measurements are **NOT TESTED**. This qualifies for the **explicit runtime-unavailable block** in P01 acceptance, not a live PASS. No GPU/image-generation capability is claimed.

### Completion
- **Implemented:** P01 offline foundation only, with host diagnostic entrypoint ready for local Forge verification.
- **Tests:** 19 deterministic offline tests PASS; actual Forge integration and real model tests NOT TESTED (runtime unavailable).
- **Branch / PR:** `phase/P01-foundation`; [PR #1](https://github.com/Neb963/forge-coherent-diffusion/pull/1); squash merge subject to final CI.
- **Blockers for live evidence:** installed Forge Neo version, actual Anima Q4 GGUF path/encode, SDXL/Anima GPU inference unavailable. Preserve user's existing loader; do not replace it or pretend probes succeeded.
- **Next:** P02 or P03 are dependency-ready. Stop after P01; do not start either in this chat.

## Update protocol
1. Before coding: select/resume current phase, inspect code and write the detailed plan here.
2. During work: check off tasks and add concrete obstacles.
3. After testing: record test commands, PASS/FAIL and actual GPU/hardware configurations.
4. After publishing: mark DONE only if P01 acceptance met, unlock P02 and P03, and STOP.

## Active phase — P02: Integrated regional editor and CRF workflow (2026-10-24)

**Status:** DONE (offline acceptance); PR #2 merged as `2ee721a`; P02 follow-up PR #3 verifies numerical influence and advisories. **Live Forge/SDXL/Anima/Q4 tests are P07 only.**

### Actual repository inspection / acceptance boundaries
- P01 supplies a strict Python CRF v1 validator, bbox/polygon raster and read-only Forge `scripts.Script` diagnostic. No editor is mounted yet.
- `docs/UI_PROTOTYPE.html` contains the intended substantial interactive UI, with five initial regions, localStorage draft, layer tools, inspector and preliminary paste/export. It is NOT production CRF: its freehand `stroke` type and exported `ui` field violate `schema_v1.json`, its importer is not equivalent to Python semantic validation, and its overlays are illustrative.
- P02 must preserve prototype UX, maintain a single CRF source, make mask assets data-only/verifiable and ship a truthful editor-only Forge mount. No P03/P05 sampler or model inference claims. Live Forge runtime was unavailable during P01; retry only if exposed.

### Detailed phase plan — committed BEFORE implementation
- [x] Integrate a complete interactive, full-width editor into existing Forge `scripts.Script` without changing the read-only probe or sampling path; keep source UI available standalone for browser verification. Local, no CDN assets.
- [x] Reuse the prototype's canvas/layer/inspector functions and repair state invariants: five-region visibility while editing, box/polygon/brush/erase, locks/visibility/enabled, resize/polygon points, zoom/pan, reorder/duplicate/delete, numeric geometry, prompt and adapter edits, independent undo/redo and autosave.
- [x] Bring export/paste/JSON file import under CRF v1: no `ui` or `stroke` in exported schema; preserve null/missing/empty prompts and persistent editor properties using CRF `regions[].extensions`; validate before application, explicit preview/replace/merge/conflict mapping, aspect policy and safe failure.
- [x] Raster brush regions must use real bounded mask assets and ZIP archive import/export (hash/path/size/mime/duplicate/decompression checks); no fake generated masks or silent data loss; reject unsafe/missing assets transactionally.
- [x] Add deterministic correct `weighted` / `priority` planned mask previews, numerical inspection and coverage warnings with separate visible versus enabled semantics; clearly label unconnected generation/capability controls.
- [x] Add browser-based UI interaction tests including five visible regions, undo/reload/CRF exports/imports, bad inputs, brush and masks, overlaps, sibling isolation and aspect mapping; Python ZIP/CRF semantic tests and CI.
- [x] Run offline CI tests on published commit, verify branch/PR, record actual evidence. Final acceptance run `40137761285` PASS at `2416d9d`; no installed Forge runtime tested.

### P02 implementation evidence and acceptance boundary (2026-10-24)
- **Canonical publish checkpoint:** full-width top-level **Coherent Diffusion** tab registered with Forge Neo's documented `script_callbacks.on_ui_tabs` API, with P01 Script diagnostics preserved in txt2img and no diffusion callbacks/patches. Editor source is self-contained `coherent_diffusion/editor/index.html` (no remote assets); independently verified inside a real Gradio 4 UI using a minimal Forge callback stub, not a running Forge Neo application.
- **Editor controls implemented:** five initial persistent visible regions; box, polygon, brush/eraser, select/move/resize, canvas and inspector edits, visibility/lock/enabled, layer reorder/duplicate/delete, prompt/negative/LoRA-reference editing, zoom/pan, local autosave with quota warning, explicit new-scene confirmation and per-scene undo/redo. No sampler backend or LoRA activation.
- **CRF interoperability:** strict fenced/raw v1 JSON validation with schema-compatible extension metadata; separate mask files in portable `.crf.zip` with PNG SHA-256, path, dimension, CRC and bounded ZIP central-directory/size checks. Browser conformance compared with all 11 valid / 28 invalid Python v1 fixtures (mask manifest is semantically valid CRF but intentionally refused without binary ZIP contents); transactional replace/merge and aspect policy.
- **Security/negative cases:** Python `load_crf_zip` rejects bad checksums, missing/extra entries, duplicate/path-traversal names, size mismatch/invalid headers; browser rejects bad ZIP and leaves the current scene unchanged. Browser WebP mask editing is not supported; use PNG.
- **CI evidence:** commit `d13cd05f8599338fd1388e063ac9091cbf74e273`, [GitHub Actions run 40135683112](https://github.com/Neb963/forge-coherent-diffusion/actions/runs/40135683112) **SUCCESS**, Python 3.11 **30 unittest PASS**, Chromium editor interaction regression **PASS** (including 25 undo/redo actions, polygon/box/numeric/aspect-merge, masks, CRF Python round-trip), and isolated real Gradio 4.40 tab/iframe interaction **PASS**. Gradio smoke uses a compatibility-pinned FastAPI/Starlette environment; not a claim about user's installed Forge packages.
- **Unresolved acceptance:** actual Forge Neo extension discovery, generated Forge page tab render, user's browser CSP/localStorage and target-device large mask responsiveness **NOT TESTED** because installed Forge, GPU and user's browser are unavailable here. Region influence is a local approximate display, not P06 effective conditioning compiler coefficients. This is P07 verification, not a P02 acceptance condition; do not invent such evidence.
- **Publication:** branch `phase/P02-editor`, [draft PR #2](https://github.com/Neb963/forge-coherent-diffusion/pull/2). Preserve the checkpoint; squash merge after remaining **offline P02** acceptance passes. **Do not start P03 in this chat**.


### P02 resumption plan — 2026-10-24 (pre-implementation)
Inspection of the live branch at `e090bc7`: the editor and ZIP/CRF path exist, but UX-04 is not proven. `paintWeights()` only draws a single aggregate tinted alpha; the priority branch drops every lower layer on any foreground hit, and its test checks merely one nonzero pixel. No per-region normalized coefficient inspector, uncovered-area feedback, or incompatible overlapping LoRA warnings is present. P02 remains the **only** phase in scope.

- [x] Replace the approximate single-channel visualization with a deterministic, model-independent **planned mixing** sampler returning per-region and base coefficients at any normalized point. Weighted: normalized nonnegative `mask * generation_weight` plus base. Priority: stable top-first fractional occlusion (mask coverage, not display visibility), then normalized weighted contributions. Keep display opacity and the `visible` flag out of coefficient math. Mask raster sampling and feather widths remain documented approximations, not P06 backend results.
- [x] Show inspectable point coefficients in the existing canvas footer, render a color-mixed influence map, and warn about uncovered / near-zero base areas and overlapping differing LoRA references or requested isolation. Make these warnings clearly advisory; no false backend precision claims.
- [x] Extend Chromium regression with numeric weighted / priority reference cases, layer visibility versus enabled, zero-weight and base-only conditions, fixed tie-break order, coverage warnings, and overlapping LoRA warning. Verify selected edits and PNG cache behavior remain safe.
- [x] Publish candidate, run and inspect required Python / Chromium / isolated Gradio CI checks on exact candidate; update evidence. **Do not perform a real Forge smoke before P07.** If offline P02 tests pass, accept and merge regardless of live Forge availability; do not start P03 in this chat.

### P02 final offline acceptance — 2026-10-25 (plan before code)

**Baseline:** P02 PR #2 was squash-merged at `2ee721a`, despite remaining P02 items still explicitly unchecked in this tracker. On merged main, `plannedInfluenceAt` computes normalized weighted and fractional-priority shares with index-stable tie-breaking, and `paintWeights` draws the composite, but only one nonzero-alpha pixel was tested. The footer samples the center, not the user's pointer; near-zero base warnings and unsupported isolation warnings are missing. Resume **P02 only**, rather than starting P03.

- [x] Make planned coefficient sampling inspectable at actual canvas/thumbnail pointer coordinates, keeping view visibility separate from generation-enabled state; retain UI-only approximation labeling. Clarify fractional priority, weight-zero and deterministic tie behavior.
- [x] Warn distinctly on uncovered points and base share below a documented threshold; flag incompatible overlapping adapter intents and requested prediction isolation as advisory, not activated capabilities; avoid false alerts from invisible-only display toggles or unavailable PNG cache pixels.
- [x] Add Chromium fixture-based deterministic reference cases: normalized weighted and priority sums/shares, feathered top-layer partial occlusion, stable equal-z document tie, base-only and disabled/zero-weight modes, visibility independence, coefficient point readout, coverage/LoRA warnings, PNG preview cache, and no errors.
- [x] Update P02 editor documentation and tracker accurately; publish candidate on this phase branch with a PR, run exact-head Python/Chromium/isolated Gradio CI, inspect failures and repair as needed. Mark P02 DONE only on credible offline acceptance evidence; actual Forge/model/GPU checks remain P07-only.

### P02 acceptance / publication — 2026-10-25

- **Behavior:** deterministic planned `weighted` and fractional `priority` mask shares, index-stable equal-z ties, cursor-point readout and a color-mixed preview. Display `visible` has no conditioning effect; `enabled=false` and zero-weight do. Coverage/low-base and LoRA overlap/prediction-isolation warnings are advisory only.
- **Regression:** new `tests/influence_browser.cjs` validates numerical coefficients, feathered partial occlusion, visibility, enabled/zero weight, base-only and uncovered authoring states, differing/equal LoRA references, unavailable isolation, cursor point position and asynchronously decoded PNG masks.
- **Verified candidate:** [GitHub Actions 40137761285](https://github.com/Neb963/forge-coherent-diffusion/actions/runs/40137761285) on `2416d9dd0720bd2c3ccb2e23239724873c8f2efb` completed **SUCCESS**: Python `cpu-tests` PASS (30 tests), original Chromium editor browser PASS, new influence numerical Chromium PASS, isolated Gradio 4 tab/iframe PASS. Earlier failure in `40137558299` was fixture-injection timing: editor `pagehide` autosaved before reload; corrected by setting fixture before editor initialization, then retested.
- **Acceptance boundary:** P02 fully accepted offline. No Forge Neo host run, model weights, GPU, low-VRAM sampler, actual conditioning or LoRA activation tested; those remain P03–P07 as defined. This PR only closes P02.
- **Publication:** phase branch `phase/P02-influence-acceptance`, [PR #3](https://github.com/Neb963/forge-coherent-diffusion/pull/3), prepared for squash merge. P03 becomes next eligible phase after accepted merge.

## Active phase — P03: SDXL synchronized tiled sampler (2026-10-25)

**Status:** DONE — offline P03 acceptance (real Forge/SDXL/GPU deferred to P07). Branch `phase/P03-sdxl-sync`. Baseline main `6adc1bdc7fd2a284fe12f738b03b258e849c5ed8` (P02 accepted and merged).

### Repository/source inspection before implementation
- Current production code contains only P01's scalar CPU `sync_reference.py`, CRF/region editor and read-only diagnostic Script. No torch tensor sampler, SDXL denoiser bridge, coherent generation invocation or full-canvas memory coordinator exists. The P02 editor deliberately does not call inference.
- Reviewed `docs/DESIGN.md` §§5–7, 11–12, `docs/P01_RUNTIME_PROBES.md`, current Forge Neo `backend/sampling/sampling_function.py`, `backend/diffusion_engine/sdxl.py`, `backend/patcher/vae.py` and MultiDiffusion for the *source-level* interfaces. Forge exposes `sampling_function_inner` (returns CFG-combined denoised prediction), `sampling_prepare/cleanup`, `unet.model`, `vae.decode_tiled`. Actual user's Forge remains outside scope until P07.
- P03 will provide a bounded offline SDXL-compatible Euler (ODE) path with explicit sigma/model capability preflight. It will **not** imply arbitrary Forge sampler, SDXL accelerated/RF, Anima, P04 semantic coherence, P06 regional conditioning or real VRAM quality verification.

### Concrete implementation plan — recorded before code
- [x] Introduce real torch tensor tile geometry/planner in image and latent coordinates, overlapping positive write weights, optional clipped context halo, deterministic ordering and full coverage preflight; fail closed on misalignment, size/memory-limit or unsupported latent geometry.
- [x] Implement CPU-resident single seeded global Gaussian field and full-canvas synchronized `x0` accumulation: every tile predicts from a frozen full-state snapshot, transfer only its halo crop to the requested inference device, accumulate on CPU in float32, then perform **one** Euler update over the whole canvas per sigma interval; no per-tile noise/solver/step history. Validate tensor shape/finiteness and time/schedule semantics.
- [x] Build an explicit Forge Neo SDXL adapter with source-backed `sampling_function_inner`, conditioning contract, memory-management prepare/cleanup and host tiled VAE decode (through Forge's `vae.decode_tiled` and latent scaling). Fail on unsupported inpainting/RF/predictors/active ControlNet/regional state rather than silently discarding capabilities; never alter existing model/encoder or patch global Forge functions.
- [x] Ensure interruption, OOM, bad prediction and failures release per-window/device state and call adapter cleanup via `finally`; retain untouched input on failure and expose bounded step/window/CPU-residency telemetry with no fabricated memory numbers.
- [x] Add offline torch-backed synthetic fixtures for tile geometry and border coverage, halo correctness, frozen-state stability, order independence, reference untiled equality (local pointwise predictor), one solver update per step, multi-step global sigmas, seeded determinism, failure/interrupt and cleanup, fake Forge SDXL interface + decode contracts; add CPU tests to CI with pinned lightweight CPU torch install.
- [x] Document precise supported settings / deferred P04–P07 integration and run candidate-head CI (Python/torch and existing Chromium + isolated Gradio). Publish PR, merge only when all offline P03 acceptance evidence passes; update this tracker accurately. **Do not start P04.**

### P03 implementation, verification and publication (2026-10-25)

- **Implementation:** `coherent_diffusion/scheduler/sdxl_sync.py` adds an image-pixel/latent-coordinate tile planner, clipped halo reads, positive cosine write taper, CPU float32 global field, single seeded full-image noise, deterministic canonical tile order, one global Euler ODE update per sigma interval and strict tensor/schedule/memory validation. `RunResult` reports effective tile geometry and *estimated* CPU and per-window bytes; these are not GPU/CPU measured peaks.
- **SDXL host interface:** `coherent_diffusion/forge_bridge/sdxl.py` uses existing Forge SDXL encoder/`compile_conditions`, `sampling_function_inner`, `sampling_prepare/cleanup`, model precision and `vae.decode_tiled`, without global sampler patches, model replacement or Q4 encoder mutation. Unsupported Anima/refiner/flow/inpaint, custom hooks, controlnets and unknown VAE geometry fail closed.
- **Recovery:** Interruption, bad tensors, nonfinite predictions, preparation failure and host-recognized OOM cause scoped cleanup. Limited retile retries replay from the same global seed and fresh solver state with smaller tiles; no partial step advancement. Incorrect unsupported schedules are rejected rather than substituted. No generation UI is activated in P03.
- **Tests, exact code checkpoint:** [GitHub Actions run 40139751481](https://github.com/Neb963/forge-coherent-diffusion/actions/runs/40139751481) for `cdd98af40d273c37137a6aac504335d2283e6557` **SUCCESS**: 22 *executed* CPU PyTorch synthetic/Forge-stub SDXL tests PASS, existing Python CRF/probe archive suite PASS (30 substantive foundation/editor tests), Chromium editor + numerical-influence browser regressions PASS, isolated Gradio tab/iframe smoke PASS. The standard Python job skips torch tests on purpose; the dedicated pinned CPU-torch job executes them.
- **Real evidence boundary:** Forge Neo installed environment, actual SDXL output, real sigma-profile compatibility, model loading, GPU peak VRAM, decoder behavior, low-VRAM device-specific fallback and visual seam/one-sun quality remain **NOT TESTED** and reserved for P07; P04 adds offline global coherence algorithms. No actual image-generation success is asserted.
- **Documentation:** [docs/P03_SDXL_SAMPLER.md](docs/P03_SDXL_SAMPLER.md). **Publication:** [PR #4](https://github.com/Neb963/forge-coherent-diffusion/pull/4), branch `phase/P03-sdxl-sync`; accepted offline, squash merge after status/documentation-only head checks. **Next eligible phase: P04. Stop after P03.**
