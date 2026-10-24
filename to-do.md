# TODO — phase tracker

Read `PHASES.md` first. **Exactly one phase per chat.** Only mark DONE with tested evidence. On beginning a phase, replace the active-plan template below with your own phase-specific plan based on actual repository code; keep the overall roadmap intact.

| Phase | Status | Depends on |
|---|---|---|
| P01 — Foundation, probes, CRF, tests | **READY** | — |
| P02 — Regional editor and CRF UI | BLOCKED | P01 |
| P03 — SDXL synchronized tiled generation | BLOCKED | P01 |
| P04 — SDXL global coherence | BLOCKED | P03 |
| P05 — Anima + Q4 GGUF parity | BLOCKED | P01, P03, P04 findings |
| P06 — Regional prompts, LoRAs, controls | BLOCKED | P02, P04, P05 |
| P07 — Final package and simple manual guide | BLOCKED | P01–P06 |

## Active phase — P01 (not started)
**Status:** READY.

### Plan (coding agent writes this *before* implementation)
- [ ] Inspect repository, current Forge Neo APIs, attached specs and upstream source.
- [ ] Replace this placeholder with a concrete, bounded implementation checklist and tests.

### Work log / evidence
- None yet. Do not infer source-read evidence is successful live inference.

### Completion
- **Implemented:** none.
- **Tests:** none.
- **Branch / PR / commit:** none.
- **Blockers:** none known.
- **Next:** P01.

## Update protocol
1. Before coding: select/resume current phase, inspect code and write the detailed plan here.
2. During work: check off finished tasks and add any concrete obstacles.
3. After testing: record test commands, PASS/FAIL and any actual GPU/hardware configurations.
4. After publishing: mark a phase DONE **only if acceptance is met**, unlock the next dependency-ready phase, and STOP.
