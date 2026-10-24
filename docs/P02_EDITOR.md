# P02 regional editor — implementation and verification

## Scope and boundaries

P02 provides a local, full-scene composition editor, not image generation. The editor is bundled as `coherent_diffusion/editor/index.html` and mounted through Forge's documented `script_callbacks.on_ui_tabs` top-level tab hook in `scripts/coherent_diffusion.py`. It does not patch Forge sampling, text encoding, LoRA or ControlNet. The P01 diagnostic panel is retained. Both SDXL and Anima settings are composition preferences; neither model is invoked.

The UI derives from the committed `docs/UI_PROTOTYPE.html` design and is self-contained (no CDN or remote images). The five default visible layers are Sky, Ocean, Beach, Clouds and Sun. Layer visibility (`visible`) is purely visual; `enabled` denotes future generation intent. Geometry is normalized top-left, independent of viewport zoom. Layer edit, box/polygon/brush/eraser, inspector, prompts, LoRA **references only**, undo/redo and draft save are browser-side operations.

## Import/export contract

- Vector scenes: strict CRF v1 JSON, exported as `.crf.json`. Raster/brush scenes: `.crf.zip` with `composition.json`, `manifest.json` and `masks/*.png`. The editor calculates SHA-256 of actual PNG data and writes matching dimensions/paths. If masks exist, copying JSON alone is deliberately blocked.
- The editor keeps stroke control points in inert CRF geometry extension metadata (`cd.editable_stroke`) while exporting a real raster mask asset, so the JSON is schema-compatible, and the original brush path can be reconstructed from its ZIP. Painting an imported image mask modifies only its bitmap and can be undone.
- Import accepts one fenced/raw JSON document or local JSON/ZIP. The browser validates data fields and references, rejects duplicate JSON object keys, future versions, overlapping invalid polygons, unsafe asset names, missing masks, excessive size, invalid SHA/CRC/dimensions and duplicate ZIP entries. A preview is displayed before an explicit Replace/Merge action. Merge renames conflicting region/LoRA/entity IDs and preserves current global scene; aspect mismatch requires an explicit relative-coordinate or vector-only letterbox choice.
- Python `coherent_diffusion.crf.assets.load_crf_zip` independently checks archives without extraction or external requests. Always pass the returned document through `parse_validate` and do not treat assets as trusted executable content. P06 must still decode and normalize masks for *generation*. No generated masks are silently substituted at inference time.

Browser `localStorage` holds the draft, including binary mask data. Saving very large compositions is limited by the browser's storage quota; prefer exporting the ZIP before clearing browser data. File import/export works offline.

## Tests

```bash
python -m pip install -r requirements.txt
python -m compileall -q coherent_diffusion scripts tests
python -m unittest discover -s tests -v
npm install --no-save playwright@1.56.1
npx playwright install chromium
node tests/editor_browser.cjs
# Requires the isolated Gradio 4.40 compatibility environment used in CI:
node tests/gradio_browser.cjs
```

CI runs the Python and Chromium checks in separate jobs. The browser test loads the standalone editor at a real local HTTP origin, confirms initial five-region visibility, state edits, round-trips JSON and PNG ZIP through the Python validators, autosave/reload, import safety, undo/redo, influence sample, geometry, and CRF conformance. It also launches an actual Gradio 4.40 Blocks instance using the registered Forge tab callback with mocked Forge modules and drives the UI via Chromium. It does **not** run a real Forge Neo application or inspect GPU/model inference.

## Optional local Forge smoke (not proved by GitHub Actions)

1. Install the repository under `Forge Neo/extensions/coherent-diffusion/` and restart Forge.
2. Open the **Coherent Diffusion** top-level tab and verify its responsive full-width editor appears (rather than an empty iframe or a security-policy error); then verify P01 diagnostics in txt2img remain available.
3. Edit one of five overlapping shapes, brush and erase without hiding sibling regions; edit prompt/LoRA reference. Export JSON/ZIP. Refresh; verify draft reload. Import your file using Validate & preview and Apply; check geometry and text.
4. Record installed Forge/Gradio versions, browser, any traceback or CSP errors, and whether all controls respond. This smoke is **NOT TESTED** without the user's Forge host. Do not infer it from mocked Gradio or standalone Chromium.
5. Do not run image generation through this extension: P03/P05/P06 implement samplers/conditioning. The Generate action only explains its unavailable status.

## Current limitations / proof obligations

- Browser editor integration in **actual Forge Neo** (especially Gradio iframe `srcdoc` and local-storage policy): pending real host verification.
- Editor influence map computes lightweight normalized relative masks and priority locally; it is not a sampler-derived conditioning heatmap, and tile view remains indicative until P03/P06.
- PNG masks are supported in the browser archive path. The Python archive validator also accepts WebP headers, but browser re-edit/import of WebP is not validated. This is not a promise of WebP interoperability.
- Large-canvas/large-number-of-masks responsiveness and local quota on the user's browser remain to be measured. P02 cannot make a tested VRAM claim.
- Full regional attention, LoRA activation/isolation, model/Q4 encode, ControlNet, actual output, OOM cleanup and low-VRAM GPU acceptance are later phases.

## Numerical influence-preview contract (P02 only)

The `Influence` view samples a **planned, UI-only** mask mix at 128×96 positions; it is not attention attribution or denoiser/LoRA output. Hover the canvas to inspect the normalized point and each contributing region's percentage in the footer (hover the readout for all contributors). UI overlay visibility and display opacity do **not** modify planned generation coefficients; `enabled` and `blend.weight=0` exclude them.

- **Weighted:** sample editable shapes and masks' estimated feathered coverage in [0,1], multiply by nonnegative weights, include `global.base_weight`, then normalize. A base-only sample reads 100% base.
- **Priority:** sort by descending `z_index`, with later CRF document order winning ties; multiply each layer's raw coverage by the fraction not occluded by higher masks, then normalize against base. Feathering leaves partially exposed lower layers.
- Approximation limits: polygon boundaries use point-in-polygon and edge-distance feather; PNG luma/alpha is downsampled to 128×96; still-loading PNGs are indicated as pending, not counted as uncovered. Backend-accurate masks and conditioning remain P06.
- The preview warns about missing coverage, base coefficients below 5%, and differing overlapping LoRA intent references. Requested prediction-isolation is advisory **only**; no LoRA adapter is activated or isolated in P02.
- `tests/influence_browser.cjs` numerically checks weighted/priority shares, fractional feather, stable ties, hover, visibility, enabled/zero weight, coverage, adapter warnings, and asynchronous PNG previews. CI runs this separately from the full editor workflow.

## Latest isolated acceptance evidence

- [GitHub Actions 40135683112](https://github.com/Neb963/forge-coherent-diffusion/actions/runs/40135683112), commit `d13cd05f`: Python 3.11 **30 tests PASS**; browser editor regression **PASS**; actual Gradio 4.40 top-level tab interaction **PASS**. The isolated Gradio smoke pins compatible FastAPI/Starlette to avoid an unrelated Gradio 4 runtime mismatch.
- **P02 final offline acceptance:** [GitHub Actions 40137761285](https://github.com/Neb963/forge-coherent-diffusion/actions/runs/40137761285), commit `2416d9d`: Python CPU unit tests **PASS** (30), original Chromium editor interaction suite **PASS**, new numerical influence suite **PASS**, isolated Gradio 4 iframe interaction **PASS**. PR #2 was merged earlier; PR #3 closes its outstanding numerical acceptance.
- The user's installed Forge Neo, Gradio/FastAPI patchset, GPU, model loader and browser policies were not accessible. *Live Forge* display and output remain **NOT TESTED** until P07.
