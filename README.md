# Coherent Diffusion — Forge Neo

Native high-resolution coherent txt2img for **SDXL and Anima**, with a multilayer region editor and CRF prompt interchange.

**P01 status:** read-only diagnostics and offline foundations. **This does not yet generate images**, route prompts or implement the editor. The HTML in docs is a design reference only. See [PHASES.md](PHASES.md) and [to-do.md](to-do.md).

## Install P01 diagnostic panel

Place this repository folder at `Forge Neo/extensions/coherent-diffusion/` and restart Forge Neo. In txt2img expand **Coherent Diffusion — diagnostics only**. Clicking **Inspect loaded model** reports non-sensitive capability/encoder metadata about an already-loaded model. Optional process capture does not change the sampler.

The P01 CRF validator uses Python `jsonschema==4.25.1`; install `python -m pip install -r requirements.txt` to validate CRF outside Forge. The diagnostic panel itself has no additional runtime dependency beyond Forge's Gradio/Python stack.

For source-level API notes, installation diagnostics and Q4 GGUF limitations see [docs/P01_RUNTIME_PROBES.md](docs/P01_RUNTIME_PROBES.md). **A real Forge smoke test has not been executed by CI.** Nothing in P01 replaces the installed Anima text-encoder loader.

## Offline checks

```bash
python -m pip install -r requirements.txt
python -m compileall -q coherent_diffusion scripts tests
python -m unittest discover -s tests -v
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

CPU tile reference and invariant checks are **not** a diffusion backend, and the benchmark metadata recorder never fabricates performance results. P03/P05/P06 are responsible for the model-specific generation and conditioning.

## Agent workflow

Follow [AGENTS.md](AGENTS.md), [PHASES.md](PHASES.md) and [docs/DESIGN.md](docs/DESIGN.md). Exactly one phase per chat; first plan in `to-do.md`, then implement, verify, publish and stop.
