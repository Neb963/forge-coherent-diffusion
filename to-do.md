# TODO — phase tracker

Read `PHASES.md` first. **Exactly one phase per chat.** Mark DONE only with evidence.

| Phase | Status | Depends on |
|---|---|---|
| P01 — Foundation, probes, CRF, tests | **IN PROGRESS** | — |
| P02 — Regional editor and CRF UI | BLOCKED | P01 |
| P03 — SDXL synchronized tiled generation | BLOCKED | P01 |
| P04 — SDXL global coherence | BLOCKED | P03 |
| P05 — Anima + Q4 GGUF parity | BLOCKED | P01, P03, P04 findings |
| P06 — Regional prompts, LoRAs, controls | BLOCKED | P02, P04, P05 |
| P07 — Final package and simple manual guide | BLOCKED | P01–P06 |

## Active phase — P01: Foundation, model probes, CRF, benchmark harness
**Status:** IN PROGRESS; branch `phase/P01-foundation`.

### Pre-implementation inspection (2026-10-24)
- Main contains specs and an HTML reference only; no extension, dependencies or tests.
- Forge Neo source inspected at `Haoming02/sd-webui-forge-classic` branch `neo`, upstream commit `0c388f8b26d26d475c2c7104aea4eafdf7bbb3e7` (2026-10-23). Relevant host interfaces: `modules/scripts.py` Script/UI/process/postprocess, `modules/script_callbacks.py` callbacks, `backend/diffusion_engine/anima.py`, `backend/text_processing/anima_engine.py`, `backend/loader_gguf.py`. This is **source inspection only**, not a check of the user's installed Forge or GGUF loader. Never replace encoder path.
- No Forge installation or GPU/model runtime is exposed in this execution environment: record actual runtime probing as NOT TESTED. Offline validation must still run.

### Concrete implementation plan — written before code
- [ ] Create a minimal loadable Forge `scripts/` Script with a read-only diagnostics panel, graceful no-model/error states, scoped lifecycle observation (no persistent monkey patches, no generation override).
- [ ] Add host probes that report model/encoder object types and qualified paths, `forge_objects` patchers, conditioning entry-point, sampler/callback/position support as supported/unknown; explicitly report inability to **prove** Q4 GGUF from a file extension or object name alone. Include compatible runtime source revision and cleanup evidence.
- [ ] Define portable CRF v1 Draft 2020-12 schema for exact example format; conservative bounds and extensible namespace; strict JSON/fenced parsing with duplicate-key/NaN rejection, geometric/ID/overlap/asset semantic checks and explicit missing/empty/null prompt handling; no external execution/download.
- [ ] Deterministic CPU-only bbox/polygon masks with documented edge/pixel convention, self-intersection rejection, positive coverage; canonical JSON round-trips and at least 10 valid/20 invalid conformance fixture documents.
- [ ] Implement small synchronized tile prediction reference harness: frozen global snapshot, positive-coverage weighted accumulator, one global update and order-invariance tests (explicitly not a production sampler).
- [ ] Add bounded benchmark metadata recorder (no fake timing/memory results), rapid unittest/schema/compile checks in GitHub Actions, short run/test/probe documentation.
- [ ] Execute deterministic tests and inspect GitHub Actions at the exact commit. Run real Forge smoke only if a matching host is actually accessible; report it NOT TESTED otherwise.
- [ ] Update status/evidence in `to-do.md`, publish code on P01 PR; squash-merge only if acceptance is satisfied. Stop without beginning P02.

### Work log / evidence
- Source inspection and phase branch creation done. No generation/host runtime claim.

### Completion
- **Implemented:** pending.
- **Tests:** pending.
- **Branch / PR / commit:** `phase/P01-foundation`; PR pending.
- **Blockers:** matching local Forge runtime, actual Anima Q4 GGUF encode, SDXL/Anima GPU generation not accessible here. P01 permits explicit integration smoke block; these are P03/P05 live obligations.
- **Next:** finish P01.

## Update protocol
1. Before coding: select/resume current phase, inspect code and write the detailed plan here.
2. During work: check off tasks and add concrete obstacles.
3. After testing: record test commands, PASS/FAIL and actual GPU/hardware configurations.
4. After publishing: mark DONE only if P01 acceptance met, unlock P02 and P03, and STOP.
