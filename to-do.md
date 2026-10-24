# TODO — phase tracker

Read `PHASES.md` first. **Exactly one phase per chat.** Mark DONE only with evidence.

| Phase | Status | Depends on |
|---|---|---|
| P01 — Foundation, probes, CRF, tests | **DONE (offline acceptance; live Forge blocked)** | — |
| P02 — Regional editor and CRF UI | IN PROGRESS | P01 |
| P03 — SDXL synchronized tiled generation | READY | P01 |
| P04 — SDXL global coherence | BLOCKED | P03 |
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

**Status:** IN PROGRESS; P01 merged in PR #1; branch `phase/P02-editor`.

### Actual repository inspection / acceptance boundaries
- P01 supplies a strict Python CRF v1 validator, bbox/polygon raster and read-only Forge `scripts.Script` diagnostic. No editor is mounted yet.
- `docs/UI_PROTOTYPE.html` contains the intended substantial interactive UI, with five initial regions, localStorage draft, layer tools, inspector and preliminary paste/export. It is NOT production CRF: its freehand `stroke` type and exported `ui` field violate `schema_v1.json`, its importer is not equivalent to Python semantic validation, and its overlays are illustrative.
- P02 must preserve prototype UX, maintain a single CRF source, make mask assets data-only/verifiable and ship a truthful editor-only Forge mount. No P03/P05 sampler or model inference claims. Live Forge runtime was unavailable during P01; retry only if exposed.

### Detailed phase plan — committed BEFORE implementation
- [ ] Integrate a complete interactive, full-width editor into existing Forge `scripts.Script` without changing the read-only probe or sampling path; keep source UI available standalone for browser verification. Local, no CDN assets.
- [ ] Reuse the prototype's canvas/layer/inspector functions and repair state invariants: five-region visibility while editing, box/polygon/brush/erase, locks/visibility/enabled, resize/polygon points, zoom/pan, reorder/duplicate/delete, numeric geometry, prompt and adapter edits, independent undo/redo and autosave.
- [ ] Bring export/paste/JSON file import under CRF v1: no `ui` or `stroke` in exported schema; preserve null/missing/empty prompts and persistent editor properties using CRF `regions[].extensions`; validate before application, explicit preview/replace/merge/conflict mapping, aspect policy and safe failure.
- [ ] Raster brush regions must use real bounded mask assets and ZIP archive import/export (hash/path/size/mime/duplicate/decompression checks); no fake generated masks or silent data loss; reject unsafe/missing assets transactionally.
- [ ] Add deterministic correct `weighted` / `priority` effective mask previews and coverage warnings with separate visible versus enabled semantics; clearly label unconnected generation/capability controls.
- [ ] Add browser-based UI interaction tests including five visible regions, undo/reload/CRF exports/imports, bad inputs, brush and masks, overlaps, sibling isolation and aspect mapping; Python ZIP/CRF semantic tests and CI.
- [ ] Run local/CI tests on published commit, verify repository branch and PR, record exact evidence and unavailable Forge/browser/GPU tests. Mark DONE only if P02 acceptance is demonstrated; do not advance to P03.
