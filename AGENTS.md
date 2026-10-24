# Coding agent instructions

Read `PHASES.md`, `to-do.md`, `docs/DESIGN.md`, and the UI reference before coding. `PHASES.md` groups the detailed design into seven deliverable phases; `docs/DESIGN.md` governs functionality.

**Exactly one phase per chat.** Inspect the actual repository and current status. Resume an unfinished phase; otherwise select the first dependency-ready phase. **First write that phase's concrete, checkable plan in `to-do.md`**, informed by the code and specifications. Then implement it, test it, update `to-do.md`, commit/publish, and stop.

Keep things simple: straightforward Python/JS, no speculative frameworks, no elaborate governance, no unnecessary abstractions. Use a phase branch and PR where possible; squash-merge after relevant tests pass. If merge or required offline verification is blocked, preserve work on the branch and record the exact blocker. Do not claim untested features work.

Preserve the prototype's layout and UX intent. All visible product controls must work before final acceptance; do not silently disable capabilities. SDXL and Anima, including the existing Anima Q4 GGUF text-encoder route, are first-class requirements. Never replace the user's working loader to make tests pass. Low-VRAM and full-image coherence are foundational; don't substitute independent per-tile txt2img.

**P01–P06 are accepted using offline testing only.** Use CI unit/synthetic tests, browser tests, Forge API stubs and isolated Gradio integration; do not require live Forge Neo, GPU, real checkpoint inference or user action. **Live testing happens only in P07.** Lack of the user's installed environment is not a pre-P07 blocker, and offline PASS must not be presented as successful real inference. Don't make the user repeat tests already proven. Last phase must package a **complete installable Forge Neo extension ZIP**, with minimal installation and manual-test instructions.

At the end report: phase, what works, tests run, commit/PR, blockers, and next phase. **Stop after one phase.**
