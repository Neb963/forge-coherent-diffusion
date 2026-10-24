# Coherent Diffusion — Forge Neo

Native high-resolution, compositionally coherent txt2img for **SDXL and Anima** on limited VRAM, with a multilayer regional editor, CRF prompt interchange, and optional regional LoRAs.

**Status:** planning/bootstrap only. No working generation extension has been implemented yet.

## Read first

1. [AGENTS.md](AGENTS.md) — small coding-agent protocol.
2. [PHASES.md](PHASES.md) and [TODO.md](TODO.md) — phase roadmap and current work.
3. [docs/DESIGN.md](docs/DESIGN.md) — authoritative technical/product specification.
4. [docs/UI_PROTOTYPE.html](docs/UI_PROTOTYPE.html) — high-fidelity *design reference*, not a backend.
5. [docs/research/](docs/research/) — background reports; resolve discrepancies using DESIGN.md and upstream verification.

Each chat implements **one eligible phase**; the agent first writes the phase plan in `TODO.md`, then codes, tests, updates progress, and stops. The final phase ships a complete installable Forge Neo extension ZIP and a short local test guide.

### Typical agent prompt

> Implement the next eligible phase in https://github.com/Neb963/forge-coherent-diffusion. Read AGENTS.md, PHASES.md, TODO.md and the specs. Plan that phase in TODO.md first, then implement and test it fully. Complete exactly one phase, update TODO.md, publish the changes, and stop.
