# P06 — Regional conditioning, serial LoRA evaluation and typed structural guidance

**Verification tier: offline Python 3.11/PyTorch 2.5.1 CPU, synthetic Forge API stand-ins, Chromium and isolated Gradio checks.** This code is **opt-in Python integration**, not yet wired to the editor's Generate action. Installed Forge Neo/SDXL/Anima/Q4 inference, local LoRA registry semantics, native ControlNet model activation, actual image quality/VRAM and third-party compatibility remain **NOT TESTED until P07**.

## Functional execution

`scheduler/regional.py` compiles validated CRF v1 into full-canvas latent-coefficient tensors. **Only enabled positive-weight** bbox, polygon and SHA-256-verified local PNG masks participate; UI `visible` never affects conditioning. Geometry uses global output-pixel coordinates and latent-center samples (scale 8), then read-halo window slicing; polygon signed-distance feather is in output pixels. PNG alpha (or luminance where alpha absent) is decoded with Pillow and resized to exact global latent dimensions; bitmap feather uses a bounded separable CPU box filter calibrated in output pixels; **WebP decoding** refuses rather than silently changing coverage. CRF ZIP bytes must first pass the P02 `load_crf_zip` integrity/path/size validation and then match the CRF SHA in the raster decoder. Aspect mismatch refuses rather than covertly reprojecting the scene.

**Weighted:** each region's nonnegative `mask × blend.weight` plus the explicit global `base_weight` is normalized pointwise. **Priority:** higher `z_index` first, then later document order on ties; higher masks fractionally occlude lower masks. The global base remains explicitly weighted, matching the P02 preview. Missing/zero total coverage fails before model work. Equal `PredictionState(prompt,negative,adapter set)` contributions are combined. This is an explicit model *prediction* blend and not a measured pixel/attention attribution. `global.scene_prompt` drives the single progressive anchor and direct-mode *baseline*; local region conditioning receives optional shared style, explicit local text or entity fallback for missing/null text. Deliberately empty region text suppresses entity fallback. Global scene text is not broadcast into every region.

`forge_bridge/regional.py` provides **`run_forge_regional_sdxl`** and **`run_forge_regional_anima`**, wrapping the P04/P05 existing globally synchronized quality schedulers. On each frozen global sigma state, it serially computes denoised x0 for each distinct regional prompt/adapter state against **the same tile + halo crop**, then blends x0 and hands it back for exactly one global solver update. SDXL stays 4D; Anima stays 5D with its previously implemented native instance-scoped absolute patch RoPE, constant-flow timeline and original Forge text loader. Progressive anchor remains a single global prediction (no regional masking of the anchor). Direct and progressive require explicit validated quality schedules; no per-tile independent finished images are generated.

### Two regional text modes

- **`text_mode="prediction"`** (default): source-backed full prediction blending for either SDXL or Anima; reliable semantic fallback for heterogeneous regional negative prompts and distinct denoiser adapter states, but potentially expensive. Region weights are **not** a fast attention mask internal to the model.
- **`text_mode="native_sdxl_masks"`** (SDXL only): compile each distinct prompt through existing Forge conditioning, then attach global-coordinate per-crop `mask` to Forge `sampling_function_inner` condition entries. The weighted masks are already normalized; condition strength `1/N` preserves Forge's CFG edit-strength convention. Supports one compiled condition entry per state with equal negatives, no regional LoRAs, no preexisting condition masks/patches/controls and no simultaneous structural guidance. The host can split this into multiple internal forward batches under low VRAM; **not** a promise of a single U-Net pass. Anima's five-dimensional field cannot use this two-dimensional Forge mask hook, so it remains on prediction blending.

## LoRA semantics, isolation and safety

Regional `loras[]` use only symbolic local registry names and optional SHA-256. `NativeForgeLoRAs.scoped` resolves against installed Forge `networks.available_network_aliases`, loads from the existing **local registry** and creates native denoiser patcher clones via `load_lora_for_models(model, None, ...)`; no CRF paths, URLs, arbitrary import or unapproved download. Every state switch is serialized, strictly validated, and restores original `forge_objects.unet`/`clip` even on interrupt, error or OOM. Explicit overlapping groups are evaluated separately on the same input then masked in x0; this restricts local *output contribution*, **not internal attention/context leakage** or proven artifact-free isolation.

`requested_isolation="prediction"` uses the prediction path; an `"approximate"` request currently receives the **more costly prediction-blended path** rather than an unverified inexpensive tile-level LoRA activation. This must be surfaced as a cost/capability distinction in the eventual UI. `experimental_internal`, nonzero text-encoder `te_weight`, already globally modified model patchers, Nunchaku-style shared-weight LoRA mutation, missing registry/hash and concurrent regional transactions **fail closed**. Text-encoder LoRAs are deliberately unsupported because they would alter the shared Forge encoder, including the user's Q4 GGUF configuration. This is *not* a claim of zero cross-region influence.

LoRA state changes require exclusive model ownership across the **whole Forge application**. The process-local lock protects this extension's sessions, not independent third-party generation. P07 must verify or enforce Forge-level exclusive generation ownership and hook conflicts before offering these modes live.

## Optional native control guidance

`scheduler/guidance.py` supplies `GuidanceRequest`, `GuidancePlan`, and `verified_guidance_png`. Every map is a verified local, finite CPU float32 RGB field in **full output-pixel coordinates**. Requests contain a typed `kind`, native backend name, `strength`, `start/end` sampling fractions and optionally CRF region IDs. Region masks, global coordinate slicing including read halo and progress gates are deterministic and tested; no control image is downloaded and no separate control stack is installed.

Recognized **contract names** are `sdxl_controlnet`, `sdxl_lllite`, `anima_vace` and `anima_region_controlnet`. Their presence in this table is *not proof* that the corresponding installed native extension or Anima variant is compatible. Caller must explicitly provide a **compatible native control factory** with a context-manager signature `control_factory(host, windows, state, tile)` yielding a host-native object with `get_control`. The P06 bridge injects that object into copied Forge conditioning entries for a single forward and restores all original entries on every exit. If there is no native factory, or the model/condition hook layout conflicts, the run refuses. **Structural guidance currently works only in direct mode**: progressive requests refuse because anchor-stage image/control remapping is not implemented, rather than silently ignoring guidance during the anchor. Source-backed control-factory adaptation, native ControlNet/LLLite/VACE model loading, format-specific transformer hook negotiation and real memory residency are **P07 validation/integration obligations**. No arbitrary third-party hooks are installed by default. This design deliberately avoids falsely advertising control support from a synthetic fake-native test.

## Resource limits and rejection

Default compiler limit: 1,048,576 **latent pixels**, maximum 16 distinct prediction states per window. Structural guidance: up to four maps and 4,194,304 output pixels per scene, each tensor explicitly CPU float32. The compiler preflights a conservative default 1 GiB CPU mask-workspace budget before rasterizing active regions. Additional CPU masks and per-group forward costs are on top of P03/P05 global-state/work-window residency estimates; these are **estimates, never measured VRAM**. LoRA state reloading and prediction blending may be much slower than one regular pass. Unsupported encoder or attention hooks, wrong alpha masks, malformed CRF/assets, absent coverage, unknown variants and incompatible control types are errors, not silent fallbacks.

## Opt-in API sketch

```python
from coherent_diffusion.crf.assets import load_crf_zip
from coherent_diffusion.forge_bridge.regional import (
    run_forge_regional_sdxl, run_forge_regional_anima,
)
# data = Path("composition.crf.zip").read_bytes()
# imported = load_crf_zip(data)
# doc, assets = imported.report.document, imported.assets
# quality = QualityPlan.build(...), or AnimaQualityPlan.build(...)
# result = run_forge_regional_sdxl(
#     already_loaded_supported_forge_host, quality, doc, cfg=5, seed=42,
#     asset_bytes=assets, text_mode="prediction")
# Optional structural guidance requires an explicitly validated native factory.
```

This example intentionally does not instantiate a Forge host or claim the editor generates. P07 owns production UI invocation and real Forge Neo lifecycle/LoRA/control acceptance.

## Offline test evidence

```bash
python -m pip install -r requirements.txt
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu
python -m pip install Pillow==11.3.0
python -m unittest discover -s tests -p 'test_regional_p06.py' -v
```

Tests cover numerical weighted/priority overlap, vector/bitmap feathers, stable tie order, verified image masks, entity fallback, strict negative/text semantics, read-halo global coordinates, 4D/5D grouped predictions, mock stage-aware Forge SDXL/Anima paths, LoRA rollback/hash, native SDXL condition masks and CFG, and typed/control lifecycle gates. All previous Python/Chromium editor/Gradio tests remain enabled. Review the phase tracker for **exact GitHub Actions SHA and results**. These do **not** establish real diffusion model compatibility or image quality.

**Next: P07** — final integration, live Forge Neo/model/device acceptance, native control-model adapters/host ownership, UI Generate and packaged installable ZIP with simple test/rollback instructions.
