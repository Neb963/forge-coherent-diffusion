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
```

CI runs the Python and Chromium checks in separate jobs. The browser test loads the standalone editor at a real local HTTP origin, confirms initial five-region visibility, state edits, round-trips JSON and PNG ZIP through the Python validators, autosave/reload, import safety, undo/redo, influence sample, geometry, and CRF conformance. It does **not** run a real Forge host.

## Optional local Forge smoke (not proved by GitHub Actions)

1. Install the repository under `Forge Neo/extensions/coherent-diffusion/` and restart Forge.
2. In txt2img expand **Coherent Diffusion — regional composition editor (P02)**. Verify the complete responsive embedded panel appears (rather than an empty iframe or a security-policy error); confirm existing P01 diagnostics remain available.
3. Edit one of five overlapping shapes, brush and erase without hiding sibling regions; edit prompt/LoRA reference. Export JSON/ZIP. Refresh; verify draft reload. Import your file using Validate & preview and Apply; check geometry and text.
4. Record installed Forge/Gradio versions, browser, any traceback or CSP errors, and whether all controls respond. This smoke is **NOT TESTED** without the user's Forge host. Do not infer it from mocked Gradio or standalone Chromium.
5. Do not run image generation through this extension: P03/P05/P06 implement samplers/conditioning. The Generate action only explains its unavailable status.

## Current limitations / proof obligations

- Browser editor integration in **actual Forge Neo** (especially Gradio iframe `srcdoc` and local-storage policy): pending real host verification.
- Editor influence map computes lightweight normalized relative masks and priority locally; it is not a sampler-derived conditioning heatmap, and tile view remains indicative until P03/P06.
- PNG masks are supported in the browser archive path. The Python archive validator also accepts WebP headers, but browser re-edit/import of WebP is not validated. This is not a promise of WebP interoperability.
- Large-canvas/large-number-of-masks responsiveness and local quota on the user's browser remain to be measured. P02 cannot make a tested VRAM claim.
- Full regional attention, LoRA activation/isolation, model/Q4 encode, ControlNet, actual output, OOM cleanup and low-VRAM GPU acceptance are later phases.
