# Coherent Diffusion — Forge Neo

Native high-resolution coherent txt2img for **SDXL and Anima**, with a multilayer region editor and CRF prompt interchange.

**Implementation through P06 (offline):** The integrated regional editor and CRF v1 JSON/ZIP workflow have offline browser/Gradio coverage. P03 implements an opt-in CPU-resident synchronized SDXL Euler sampler; P04 adds two opt-in SDXL quality trajectories (direct and global-first refinement), stage-specific Forge adapters and conservative CRF semantic window routing. P05 adds an opt-in 5D Anima constant-flow and global patch-RoPE bridge, separate direct/progressive execution and original Qwen/Q4 encoder ownership. P06 adds separate opt-in CRF-to-global-latent regional prompt prediction mixing for both models, SDXL native condition-mask batching under strict capabilities, serial local-registry denoiser LoRA evaluation and typed global-coordinate guidance through explicitly supplied native host control bindings. These are **source contracts and synthetic checks, not a complete installed-Forge activation**. None is activated in the Forge generation UI yet; actual model/GPU inference and supported native ControlNet/LLLite/VACE adapters are deliberately reserved for P07. See [P06 regional conditioning](docs/P06_REGIONAL.md), [P04 quality and routing](docs/P04_GLOBAL_COHERENCE.md), [PHASES.md](PHASES.md), and [to-do.md](to-do.md).

## P07 integration candidate — NOT release-ready

[P07 draft PR #8](https://github.com/Neb963/forge-coherent-diffusion/pull/8) contains an opt-in Gradio Generate panel, editor-to-Forge CRF transfer, strict model/preflight checks, native predictor-derived SDXL Euler and Anima flow sigma trajectories, CPU-resident P04/P05/P06 execution invocation, native tiled decode, cancellation and a reproducible installable ZIP **candidate** in GitHub Actions. See [INSTALL_AND_TEST.md](INSTALL_AND_TEST.md) for extraction, backup/rollback, actual SDXL/Anima/Q4 acceptance and exact known incompatibilities. Unlike P01–P06, **P07 requires real installed Forge Neo and GPU/quality evidence**. Real outputs, LoRA ownership, native control factories, and low-VRAM behavior remain unverified; this branch is not an accepted release.

### P07 live-integration repair checkpoint (still NOT accepted)

Real operator evidence confirms Anima FLOW/WAN and a GGUF text-encoder parameter marker, but *no successful Coherent Diffusion model inference has been verified yet*. The first installed build failed at dependency detection, then Forge's lazy-loaded placeholder, a standard sampler sigma metadata option, and finally a **fictitious LoRA reference in the editor's own example**. Those issues exposed missing native integration tests rather than proving GPU sampling is functional.

The current draft performs opt-in generation under Forge Neo's own FIFO GPU job lock, refreshes and reloads the selected checkpoint via native Forge methods, and restores the original model patcher objects on completion/error. Fresh editor compositions no longer contain fake active LoRAs; existing browser drafts are not silently changed. Regional denoiser LoRAs are **bypassed per generation by default**, with the number recorded in metadata and a separate explicit opt-in; the existing P06 registered LoRA patcher scopes still need real image validation. SDXL/Anima model guards accept clean native transformer-option containers and bounded finite sampler-sigma bookkeeping, but reject unnegotiated active hooks. These fixes have automated synthetic coverage; actual Q4 inference, LoRA isolation, low-memory VRAM, controls and final integrated UI acceptance remain P07 blockers. Consult [INSTALL_AND_TEST.md](INSTALL_AND_TEST.md) and the draft PR for exact live testing boundaries.

## Install P02 editor + P01 diagnostic panel

Place this repository folder at `Forge Neo/extensions/coherent-diffusion/` and restart Forge Neo. Open the full-width **Coherent Diffusion** top-level tab for the regional editor. In txt2img, the noninvasive **Coherent Diffusion** diagnostic panel remains available for model inspection. The P07 candidate adds a separate opt-in generation form; compatibility with loaded checkpoints has not yet passed live acceptance. Clicking **Inspect loaded model** reports non-sensitive capability/encoder metadata about an already-loaded model. Optional process capture does not change the sampler.

The P01 CRF validator uses Python `jsonschema==4.25.1`; install `python -m pip install -r requirements.txt` to validate CRF outside Forge. The diagnostic panel itself has no additional runtime dependency beyond Forge's Gradio/Python stack.

For P06 typed regional prompting, LoRA isolation limitations and structural-control integration gates see [docs/P06_REGIONAL.md](docs/P06_REGIONAL.md). For P05 supported Anima flow/3D RoPE/Q4 boundaries see [docs/P05_ANIMA_FLOW.md](docs/P05_ANIMA_FLOW.md). For P04 progressive/direct SDXL trajectories, explicit entity routing and experimental limits see [docs/P04_GLOBAL_COHERENCE.md](docs/P04_GLOBAL_COHERENCE.md). For the P03 tensor sampler, supported SDXL Euler contract and CPU/VRAM evidence boundaries see [docs/P03_SDXL_SAMPLER.md](docs/P03_SDXL_SAMPLER.md). For P02 import/export and browser tests see [docs/P02_EDITOR.md](docs/P02_EDITOR.md). For source-level API notes, installation diagnostics and Q4 GGUF limitations see [docs/P01_RUNTIME_PROBES.md](docs/P01_RUNTIME_PROBES.md). **Live Forge Neo testing is reserved for P07; CI does not perform it.** Nothing in P01 replaces the installed Anima text-encoder loader.

## Offline checks

```bash
python -m pip install -r requirements.txt
python -m compileall -q coherent_diffusion scripts tests
python -m unittest discover -s tests -v
# For P03–P06 tensor tests, additionally install torch 2.5.1 CPU + Pillow 11.3.0.
python -m unittest discover -s tests -p 'test_regional_p06.py' -v
npm install --no-save playwright@1.56.1
npx playwright install chromium
node tests/editor_browser.cjs
```

The CRF Draft 2020-12 schema is at `coherent_diffusion/crf/schema_v1.json`. A complete pastable sample is at `examples/one_sun.crf.json`, with 11 valid and 28 invalid conformance documents at `tests/fixtures/crf_conformance.json`.

Python API example:

```python
from pathlib import Path
from coherent_diffusion.crf import parse_validate, canonical_json
report = parse_validate(Path("examples/one_sun.crf.json").read_text())
print(report.warnings)
print(canonical_json(report.document))
```

CPU tile reference and invariant checks are **not** a diffusion backend, and the benchmark metadata recorder never fabricates performance results. P03/P04 supply offline SDXL paths, P05 supplies an opt-in source-backed Anima flow path, and P06 supplies separately callable regional conditioning and scoped LoRA execution contracts. The P07 draft integrates an opt-in Generate action but **real-host acceptance is pending**. Structural-control factories and regional LoRA isolation remain gated.

## Agent workflow

Follow [AGENTS.md](AGENTS.md), [PHASES.md](PHASES.md) and [docs/DESIGN.md](docs/DESIGN.md). Exactly one phase per chat; first plan in `to-do.md`, then implement, verify, publish and stop.
