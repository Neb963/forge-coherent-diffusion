# Native High-Resolution Regional Synthesis in Forge Neo: Technical Research, Feasibility, and Protocol Specification

## Executive Synthesis and Resolution of Foundational Architectural Decisions

Designing a native high-resolution text-to-image (`txt2img`) and spatial conditioning extension for Stable Diffusion WebUI Forge Neo (`sd-webui-forge-classic`, `neo` branch)[1](https://www.reddit.com/r/StableDiffusion/comments/1n7fd2v/introducing_sdwebuiforgeneo/) requires reconciling two distinct generative paradigms: convolutional U-Net architectures exemplified by Stable Diffusion XL (SDXL)[3](https://lzhxmu.github.io/accdiffusion/accdiffusion.html) and Diffusion Transformer (DiT) architectures exemplified by CircleStone Labs' Anima (2B Base and 3.8B v2 variants)[5](https://huggingface.co/circlestone-labs/Anima/discussions/35). Generating canvases from \(2048 \times 2048\) up to \(4096 \times 4096\) and \(8192 \times 8192\) under strict consumer VRAM budgets (4 GB to 12 GB) while maintaining global semantic coherence, local structural fidelity, and leak-free regional LoRA isolation cannot be achieved by adopting any single published paper verbatim[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf). Instead, source-code analysis of Forge Neo's ComfyUI-derived `ModelPatcher` backend[2](https://github.com/Haoming02/sd-webui-forge-classic) and empirical evaluation of extrapolation literature mandate a hybrid, multi-stage synthesis engine.

Before establishing the five formal engineering deliverables and ten research tracks, the seven foundational architectural decisions that govern the extension's design are resolved below based on verified algorithmic mechanisms and Forge Neo internal constraints.

| Architectural Decision | Selected Engineering Resolution | Primary Technical Justification & Evidence |
| --- | --- | --- |
| **1. Direct Tiled vs. Progressive Generation** | **Dual-Mode Hybrid Engine (Default: Progressive Anchor + Tiled Refinement)** | Direct patch-wise generation (MultiDiffusion) suffers from severe object repetition and loss of global topology because tiles denoise independently without a shared low-frequency structural prior[8](https://arxiv.org/pdf/2412.02099?). Progressive generation (DemoFusion / AccDiffusion v2 / FreeScale) establishes global layout at native training resolution (\(1024 \times 1024\)) in ~2–4 seconds before injecting noise-inversed skip residuals into high-resolution tiled passes[8](https://arxiv.org/pdf/2412.02099?). Direct tiled generation is retained solely as an optional fast mode for img2img/ControlNet-locked layouts[2](https://github.com/Haoming02/sd-webui-forge-classic). |
| **2. Shared vs. Separate SDXL/Anima Execution Paths** | **Unified Frontend & Latent Streamer; Bifurcated Model Patching Paths** | SDXL and Anima share the CRF parser, multilayer canvas, CPU-resident latent canvas, and tiled VAE pipeline, but require separate `ModelPatcher` hooks[10](https://note.com/198619891990/n/nf8d0cb57d57e). SDXL relies on U-Net convolutional downsampler/upsampler overrides (HiDiffusion RAD/RAU) and 2D spatial cross-attention maps[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf). Anima is a 2B/3.8B DiT utilizing 3D Rotary Position Embeddings (`3D RoPE` with scale `[1, 4, 4]`) and a timestep-aware Qwen3.5 `Semantic Connector v2`[6](https://github.com/GumGum10/forge-anima-3.8B), requiring RoPE coordinate offsetting rather than convolutional striding[16](https://arxiv.org/html/2508.15774v2). |
| **3. Global Attention vs. Efficient Approximations** | **Windowed/Tiled Self-Attention with Bijective Dilated Interaction** | Full global self-attention scales quadratically (\(O(N^2)\)) with latent token count, causing immediate OOM above \(2048 \times 2048\) on \(\le 12\text{ GB}\) GPUs[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf). Combining Multi-Scale Shifted Window Multi-Head Self-Attention (MSW-MSA) for SDXL[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf), Adaptively Rectified RoPE (AR-RoPE) tiled self-attention for Anima[16](https://arxiv.org/html/2508.15774v2), and AccDiffusion v2's position-wise bijection across dilated samples preserves long-range semantic continuity at \(O(N)\) memory scaling[3](https://lzhxmu.github.io/accdiffusion/accdiffusion.html). |
| **4. Attention-Level vs. Prediction-Level Conditioning** | **Attention-Level Masking for Text; Prediction/Shader-Level for LoRAs** | Attention-level coupling (`AttentionCouple` / `sd-forge-couple`) merges regional text keys/values inside `attn2` within a single forward pass, imposing near-zero latency overhead[13](https://www.sourcepulse.org/projects/2335762). However, prediction-level (latent) blending requires \(K\) separate forward passes for \(K\) regions[18](https://github.com/hako-mikan/sd-webui-regional-prompter). Therefore, text prompts execute via masked cross-attention, whereas prediction/feature-level blending is reserved strictly for conflicting regional LoRAs[18](https://github.com/hako-mikan/sd-webui-regional-prompter). |
| **5. Strict vs. Approximate LoRA Isolation** | **Tiered Isolation (Refined Shader / Masked LoRA Delta vs. Latent Pass)** | Because LoRAs modify both text encoder projections and U-Net/DiT linear weights globally, standard attention masking allows style and identity bleed across regions[20](https://arxiv.org/pdf/2407.02329?). The architecture implements dynamic spatial masking of on-the-fly LoRA weight deltas (\(\Delta W \cdot x \odot M_r\)) alongside MIGC++ Instance/Refined Shader gating during high-noise steps (\(t \in [1.0, 0.6]\)), falling back to sequential regional latent passes only when LoRAs heavily alter global text encoder state[2](https://github.com/Haoming02/sd-webui-forge-classic). |
| **6. Automatic Prompt Decomposition** | **Zero-Overhead Cross-Attention Binarization (No External VLM/SAM)** | Loading external segmentation (SAM) or captioning models during diffusion exhausts 4–8 GB VRAM budgets[8](https://arxiv.org/pdf/2412.02099?). During the \(1024 \times 1024\) anchor pass, the engine captures internal cross-attention maps \(M \in \mathbb{R}^{N \times M}\) and binarizes token activations against column means (\(M_{i,j} > \overline{M}_{:,j}\)) per AccDiffusion v2, automatically routing active subject tokens only to relevant high-resolution patches[8](https://arxiv.org/pdf/2412.02099?). |
| **7. Coordinates and Mask Semantics in CRF** | **Normalized Top-Left `[0,1]` Vector Primitives + `L1` / Alpha Compositing** | To guarantee resolution and aspect-ratio independence across external LLM workflows and high-resolution canvases, CRF standardizes on top-left origin `[x, y, w, h]` floats in `[0.0, 1.0]`, explicit `z_index` layering, `base_ratio` global inheritance[18](https://github.com/hako-mikan/sd-webui-regional-prompter), and dual support for pure JSON polygons and `.crf.zip` raster mask bundles. |

## Deliverable 1: Comprehensive Algorithmic and Systems Research (RESEARCH.md)

### Track 1: High-Resolution Diffusion Algorithms and 2025–2026 Extrapolation Advances

Extrapolating pretrained latent diffusion models (LDMs) and Diffusion Transformers (DiTs) beyond their native training distribution (\(1024 \times 1024\) for SDXL and Anima) triggers two fundamental failure modes: **object repetition** (generating multiple heads, limbs, or duplicate subjects) and **structural distortion** (broken perspective, local seam discontinuities, and degraded high-frequency textures)[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf). A rigorous examination of foundational and 2025–2026 training-free extrapolation methods reveals that these failures stem from distinct architectural mechanisms in U-Nets versus Transformers.

#### Foundational Extrapolation Paradigms (2023–2024)

**DemoFusion** builds upon MultiDiffusion's overlapped patch averaging by introducing three mechanisms: progressive upscaling from \(1024^2\) to target resolutions in iterative stages, skip-residual connections that inject noise-inversed intermediate latents (\(z'_t\)) from the lower-resolution pass to anchor global layout, and dilated sampling, which subsamples the high-resolution latent grid with stride \(s\) to provide a global receptive field during denoising[8](https://arxiv.org/pdf/2412.02099?). While DemoFusion prevents catastrophic structural collapse, its reliance on multiple progressive passes and dense overlapping patches (`stride=64` in latent space) makes it computationally expensive, requiring `507 seconds` at \(4096 \times 4096\) (`25x` slower than standard \(1024^2\) inference)[12](https://github.com/PRIS-CV/DemoFusion). More critically, DemoFusion suffers from persistent *small object repetition*[3](https://lzhxmu.github.io/accdiffusion/accdiffusion.html).

**AccDiffusion (ECCV 2024) and AccDiffusion v2 (2024–2025)** diagnose the exact root cause of DemoFusion's small object repetition[3](https://lzhxmu.github.io/accdiffusion/accdiffusion.html). Ablation studies demonstrate that small object repetition is the adversarial outcome of two competing forces: passing the *identical global text prompt* to every local patch (which forces each patch to independently synthesize the prompted subject) while simultaneously injecting global semantic residuals from dilated sampling[3](https://lzhxmu.github.io/accdiffusion/accdiffusion.html). Removing text prompts during high-resolution passes eliminates object repetition entirely but severely degrades fine-grained texture detail; conversely, removing skip residuals and dilated sampling causes massive large-object duplication[8](https://arxiv.org/pdf/2412.02099?). AccDiffusion v2 solves both issues without external models by decoupling the global prompt into *patch-content-aware prompts* derived from low-resolution cross-attention masks, and by introducing a *position-wise bijection function* that forces noise interaction across independent dilated sampling windows[3](https://lzhxmu.github.io/accdiffusion/accdiffusion.html).

**HiDiffusion (ECCV 2024)** investigates direct image-wise extrapolation and discovers that in U-Net architectures (SD 1.5, SD 2.1, SDXL, SDXL-Turbo), object duplication originates from **feature duplication in the deep blocks of the U-Net**[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf). When input resolution increases by \(2\times\) or \(4\times\), the fixed convolutional receptive field in `Down Block 1` and `Up Block 1` fails to compress spatial features to match the training-time dimensions of the bottleneck (`Middle Block`), causing the deep layers to tile semantic structures[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf). HiDiffusion introduces **Resolution-Aware U-Net (RAU-Net)**, which replaces the standard downsampler in `Down Block 1` with a Resolution-Aware Downsampler (RAD) and the upsampler in `Up Block 1` with a Resolution-Aware Upsampler (RAU) up to a switching timestep threshold \(T_1\) (e.g., \(T_1 = 20\) out of 50 steps, with scaling factor \(\alpha = 8\) at \(2048 \times 2048\))[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf). By dynamically resizing deep feature maps to align with native training dimensions during early structural formation and reverting to standard convolutions during late texture refinement, RAU-Net prevents feature duplication internally[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf). To eliminate the quadratic bottleneck of self-attention at high resolutions, HiDiffusion pairs RAU-Net with **Multi-Scale Shifted Window Multi-Head Self-Attention (MSW-MSA)**, achieving a `2.68x` speedup over vanilla SDXL direct inference, `1.58x` over ScaleCrafter, and `4.18x` over DemoFusion at \(2048 \times 2048\)[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf).

**ElasticDiffusion (CVPR 2024)** addresses arbitrary aspect ratios and resolutions up to \(2\times\) training scale by decomposing the diffusion score into **global content** and **local content** signals[16](https://arxiv.org/html/2508.15774v2). During inference, ElasticDiffusion computes the conditional direction on a downsampled global latent (enhanced via iterative pixel resampling governed by `resampling_steps` and percentage `new_p = 0.3`) while estimating the unconditional local texture score over overlapping patches[27](https://github.com/MoayedHajiAli/ElasticDiffusion-official). It introduces **Reduced-Resolution Guidance (RRG)** (`rrg_init_weight` between `2000` and `4000` with `cosine_scale` decay) to suppress background bleedthrough and boundary artifacts[27](https://github.com/MoayedHajiAli/ElasticDiffusion-official). While effective for aspect-ratio extrapolation (such as \(1920 \times 1080\) or \(1080 \times 1920\)), its official implementation is limited to \(2\times\) base resolution (\(2048 \times 2048\) for SDXL), exhibits high hyperparameter sensitivity across resolutions, and requires expensive resampling steps that substantially increase latency[27](https://github.com/MoayedHajiAli/ElasticDiffusion-official).

**SyncDiffusion (NeurIPS 2023)** targets global montage coherence across tiled diffusions (MultiDiffusion) by synchronizing windows via gradient descent on perceptual similarity[11](https://arxiv.org/html/2306.05178v3). At each denoising step (up to threshold `sync_thres`, e.g., the first \(N=5\) to \(50\) steps), SyncDiffusion predicts the one-step denoised image \(\hat{\mathbf{x}}_0^{(i)}\) for each window \(i\), computes the perceptual loss (LPIPS) between window \(\hat{\mathbf{x}}_0^{(i)}\) and a designated anchor window \(\hat{\mathbf{x}}_0^{(0)}\), and backpropagates gradients with weight \(w = 20\) (`sync_weight`) to update the noisy latents \(\mathbf{x}_t^{(i)}\) prior to MultiDiffusion averaging[11](https://arxiv.org/html/2306.05178v3). In human evaluations, SyncDiffusion achieved a `66.35%` coherence preference over MultiDiffusion's `33.65%`[29](https://github.com/KAIST-Visual-AI-Group/SyncDiffusion). However, backpropagating through the perceptual network and VAE decoder across all tiles incurs severe memory and compute overhead, making square \(2048 \times 2048\) generation substantially slower than 1D panoramas[31](https://replicate.com/adirik/syncdiffusion/readme). Furthermore, enforcing high perceptual similarity against a single anchor window homogenizes textures across the canvas, which is counterproductive when distinct spatial regions require contrasting prompts.

**ScaleCrafter (ICLR 2024)** enlarges the U-Net receptive field by dynamically dilating convolutional kernels at specific layers during early denoising steps[16](https://arxiv.org/html/2508.15774v2). Empirical evaluations demonstrate that while kernel dilation partially mitigates MultiDiffusion's repetitive content, it degrades local high-frequency detail, suffers from localized anatomical duplication (e.g., subjects with multiple arms or mouths), and causes GPU memory consumption to scale steeply with resolution[8](https://arxiv.org/pdf/2412.02099?).

#### 2025–2026 State-of-the-Art Extrapolation Advances

Recent literature from late 2024 through 2026 introduces critical breakthroughs that outperform earlier methods in both speed and structural fidelity. **FreeScale (2024–2025)** unlocks tuning-free synthesis up to \(8192 \times 8192\) (\(8\text{K}^2\)) by fusing multi-scale information, combining global structural priors with tailored self-attention scale fusion[9](https://arxiv.org/html/2412.09626v1). Crucially, FreeScale enables semantic mask weighting (assigning regional fusion coefficients \(\alpha\) per semantic area) to selectively boost detail in focal subjects while suppressing background repetition, outperforming DemoFusion in both visual quality and speed (`323s` vs `507s` in high-resolution image synthesis, and `3.787 min` vs `9.302 min` in video FVD benchmarks)[9](https://arxiv.org/html/2412.09626v1). Complementing this approach, **RectifiedHR (2025–2026)** targets progressive resolution scaling (\(1024^2 \rightarrow 2048^2 \rightarrow 3072^2 / 4096^2\)) by replacing slow multi-pass diffusion with **Noise Refresh**—applied at discrete switching steps such as steps 30 and 40 via \(\tilde{p}_{x_0}^t = E(\text{resize}(D(p_{x_0}^t)))\)—paired with **Energy Rectification**, which recalibrates latent feature energy degradation caused by spatial interpolation[34](https://www.alphaxiv.org/ko/abs/2407.10738).

In parallel, transformer positional extrapolation methods—including **AR-RoPE**, **Time-Aware Scaled RoPE**, **Spiral RoPE**, **DyPE**, and **UltraImage** (2025–2026)—address the unique requirements of DiT architectures[16](https://arxiv.org/html/2508.15774v2). Unlike SDXL's convolutional U-Net, DiT models such as Anima rely on Rotary Position Embeddings (RoPE)[15](https://ostris.com/docs/ai-toolkit/models/anima). When extrapolating a DiT to \(2048^2\) or \(4096^2\), standard RoPE and naive NTK-RoPE produce severe structural artifacts, jittering, and over-smoothed details because token coordinate offsets exceed the pretrained wavelength spectrum[16](https://arxiv.org/html/2508.15774v2). **Adaptively Rectified RoPE (AR-RoPE, 2025)** solves this for tiled DiT self-attention by partitioning queries into non-overlapping tiles \(\mathcal{T}_m\) while applying a piecewise linear rectification \(\rho^{(m)}\) to global keys, preserving exact relative coordinates within a local threshold \(\tau\) around the tile and linearly compressing distant key coordinates into the pretrained RoPE range[16](https://arxiv.org/html/2508.15774v2). Similarly, **Time-Aware Scaled RoPE (Next-DiT)** dynamically modulates the rotary frequency base as a function of diffusion timestep \(t\), preventing content repetition during early global layout while restoring high-frequency sharpness at late timesteps[38](https://neurips.cc/virtual/2024/poster/93994). **DyPE (2025)** and **UltraDiffEdit (2026)** further refine dynamic positional extrapolation and coarse-to-fine latent blending (\(\hat{\mathbf{z}}_t^s = \gamma_1 \bar{\mathbf{z}}_t^s + (1 - \gamma_1)\mathbf{z}_t^s\)) up to 8K resolution under bounded VRAM[37](https://arxiv.org/html/2510.20766v2).

[Interactive visualization 1 (frame_source_saved)](https://24fnp0fbhvl5vu3ysxovghygfht8tr25g7isn9aaidf2zcbwc0-h998661601.scf.usercontent.goog/gemini-code-immersive/shim.html?origin=https%3A%2F%2Fgemini.google.com&cache=1)

#### Technical Comparison Matrix of Candidate Algorithms

The table below synthesizes the capabilities, architectural dependencies, and compatibility of all evaluated candidate algorithms against the extension's core requirements.

| Algorithm / Method | Prevents Duplicated Objects? | Addresses Seams vs. Global Coherence? | Full Latent on GPU Required? | SDXL (U-Net) Compatibility | Anima (DiT) Compatibility | Relationship & Role in Proposed Engine |
| --- | --- | --- | --- | --- | --- | --- |
| **DemoFusion**[8](https://arxiv.org/pdf/2412.02099?) | **Partial** (Eliminates large duplicates via skip-residual; fails on small object repetition)[3](https://lzhxmu.github.io/accdiffusion/accdiffusion.html). | **Both** (Overlapped stride alleviates seams; dilated sampling + skip-residual anchor global structure)[12](https://github.com/PRIS-CV/DemoFusion). | **No** (Patch-wise denoising with `view_batch_size` and tiled VAE)[8](https://arxiv.org/pdf/2412.02099?). | **Native** (Designed and validated on SDXL)[12](https://github.com/PRIS-CV/DemoFusion). | **Adaptable** (Latent space skip-residuals are model-agnostic)[8](https://arxiv.org/pdf/2412.02099?). | **Core Progressive Backbone**: Provides the multi-stage noise-inversed skip-residual framework[8](https://arxiv.org/pdf/2412.02099?). |
| **AccDiffusion v2**[3](https://lzhxmu.github.io/accdiffusion/accdiffusion.html) | **Yes (Strong)** (Decouples global prompt into binary cross-attention patch prompts)[3](https://lzhxmu.github.io/accdiffusion/accdiffusion.html). | **Both** (Position-wise bijection enables noise interaction across dilated windows)[3](https://lzhxmu.github.io/accdiffusion/accdiffusion.html). | **No** (Operates patch-wise on consumer GPUs)[8](https://arxiv.org/pdf/2412.02099?). | **Native** (Validated on SDXL and SD 1.5)[8](https://arxiv.org/pdf/2412.02099?). | **Adaptable** (Cross-attention maps \(M \in \mathbb{R}^{N \times M}\) extracted from DiT cross-attn blocks). | **Mandatory Coherence Layer**: Replaces DemoFusion's naive prompt repetition and independent dilation[3](https://lzhxmu.github.io/accdiffusion/accdiffusion.html). |
| **HiDiffusion**[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf) | **Yes (Internal)** (RAU-Net resizes deep U-Net feature maps via RAD/RAU)[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf). | **Global Only** (Single-stage full-canvas pass; MSW-MSA uses shifted windows to avoid window seams)[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf). | **Yes** (Image-wise method; requires full latent in VRAM, limiting 4K on \(\le 8\text{ GB}\))[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf). | **Native** (SD 1.5, SD 2.1, SDXL, SDXL-Turbo)[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf). | **Incompatible** (RAU-Net depends on U-Net `DownBlock1` / `UpBlock1` topology)[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf). | **Primary SDXL Fast Mode (\(2048^2\))** & **Anchor Enhancer**: Accelerates \(2048^2\) passes on \(\ge 8\text{ GB}\) GPUs by \(2.68\times\)[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf). |
| **ElasticDiffusion**[26](https://arxiv.org/abs/2311.18822) | **Yes (Up to \(2\times\))** (Global score computed at base resolution prevents duplication)[27](https://github.com/MoayedHajiAli/ElasticDiffusion-official). | **Global > Seams** (RRG reduces artifacts, but high RRG causes blur/bleed and tiled VAE shows boundary steps)[27](https://github.com/MoayedHajiAli/ElasticDiffusion-official). | **No** (Supports `low_vram` and reduced `view_batch_size`)[27](https://github.com/MoayedHajiAli/ElasticDiffusion-official). | **Native** (Best results on SDXL 1.0 up to \(2048^2\))[27](https://github.com/MoayedHajiAli/ElasticDiffusion-official). | **Low** (Requires costly iterative pixel resampling and U-Net score decomposition)[16](https://arxiv.org/html/2508.15774v2). | **Superseded / Optional**: RRG concept informs CFG scaling, but \(2\times\) cap and resampling overhead make it redundant alongside AccDiffusion v2[27](https://github.com/MoayedHajiAli/ElasticDiffusion-official). |
| **SyncDiffusion**[11](https://arxiv.org/html/2306.05178v3) | **No** (Enforces perceptual similarity to anchor tile, which can clone anchor objects across panorama)[11](https://arxiv.org/html/2306.05178v3). | **Both** (MultiDiffusion averages seams; LPIPS gradient descent synchronizes global style/lighting)[11](https://arxiv.org/html/2306.05178v3). | **No**, but backprop through LPIPS + VAE per step causes high VRAM spikes[11](https://arxiv.org/html/2306.05178v3). | **Adaptable** (Originally SD 1.5/2.0; applicable to any LDM)[11](https://arxiv.org/html/2306.05178v3). | **Impractical** (Backpropagating through 2B/3.8B DiT + Qwen VAE is prohibitively slow)[31](https://replicate.com/adirik/syncdiffusion/readme). | **Excluded**: Gradient descent per step is too slow for 2D grids and conflicts with heterogeneous regional prompts[11](https://arxiv.org/html/2306.05178v3). |
| **ScaleCrafter**[16](https://arxiv.org/html/2508.15774v2) | **Partial** (Reduces repetition vs MultiDiffusion, but still yields multi-limb/eye artifacts)[9](https://arxiv.org/html/2412.09626v1). | **Global Only** (Dilated convolutions operate image-wise; degrades fine local details)[32](https://cvpr.thecvf.com/virtual/2024/poster/31490). | **Yes** (Image-wise VRAM scales heavily with resolution)[8](https://arxiv.org/pdf/2412.02099?). | **Native** (Built on SDXL convolutional layers)[32](https://cvpr.thecvf.com/virtual/2024/poster/31490). | **Incompatible** (DiTs contain no spatial 2D convolutions in transformer blocks)[32](https://cvpr.thecvf.com/virtual/2024/poster/31490). | **Excluded**: Outperformed in speed (\(1.58\times\)) and quality by HiDiffusion and FreeScale[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf). |
| **FreeScale & RectifiedHR**[9](https://arxiv.org/html/2412.09626v1) | **Yes (Up to \(8\text{K}^2\))** (Multi-scale fusion + semantic \(\alpha\) masks + noise refresh)[9](https://arxiv.org/html/2412.09626v1). | **Both** (Cosine/semantic scale fusion and energy rectification restore crisp local detail)[9](https://arxiv.org/html/2412.09626v1). | **Configurable** (Can execute in tiled windows when paired with patch streaming). | **Native** (Validated on SDXL up to \(8192^2\))[9](https://arxiv.org/html/2412.09626v1). | **High** (Noise refresh & energy rectification operate directly on latent tensors)[35](https://arxiv.org/html/2503.02537v3). | **Complementary Detail & Scaling Upgrade**: Replaces naive Gaussian filtering in DemoFusion[9](https://arxiv.org/html/2412.09626v1). |
| **AR-RoPE & Time-Aware RoPE**[16](https://arxiv.org/html/2508.15774v2) | **Yes (For DiTs)** (Prevents positional wavelength wrapping that causes DiT repetition)[16](https://arxiv.org/html/2508.15774v2). | **Both** (Preserves local tile geometry within \(\tau\) while rectifying global key offsets)[16](https://arxiv.org/html/2508.15774v2). | **No** (Designed specifically for non-overlapping query tiles \(\mathcal{T}_m\))[16](https://arxiv.org/html/2508.15774v2). | **N/A** (SDXL does not use RoPE in spatial attention). | **Native / Essential** (Directly targets 2D/3D RoPE in DiT blocks)[15](https://ostris.com/docs/ai-toolkit/models/anima). | **Primary Anima Extrapolation Engine**: Enables artifact-free tiled and extrapolated DiT synthesis[16](https://arxiv.org/html/2508.15774v2). |

### Track 4: Global Coherence and Automatic Region Intelligence

When a user provides only a global prompt (without drawing manual regions) and requests a \(3072 \times 3072\) or \(4096 \times 4096\) image, naive tiled diffusion passes the entire global prompt to every tile[3](https://lzhxmu.github.io/accdiffusion/accdiffusion.html). If the prompt specifies *"Professional landscape photograph, one small bright sun, a lighthouse on a cliff, ocean horizon"*, every sky and ocean tile receives the tokens *"sun"* and *"lighthouse"*, resulting in multiple suns and phantom lighthouses scattered across the water[3](https://lzhxmu.github.io/accdiffusion/accdiffusion.html).

#### Automatic Patch-Aware Prompt Decomposition via Cross-Attention Binarization

To eliminate object repetition automatically without requiring manual mask painting or loading a multi-gigabyte Vision-Language Model (VLM) or Segment Anything Model (SAM) into VRAM[8](https://arxiv.org/pdf/2412.02099?), the extension implements AccDiffusion v2's training-free cross-attention decomposition[8](https://arxiv.org/pdf/2412.02099?), extended with `sd-forge-couple`'s subject-count normalization heuristics[44](https://github.com/Haoming02/sd-forge-couple).

During Stage 1 (the \(1024 \times 1024\) anchor generation pass), the model synthesizes a globally coherent base latent \(z_0^{\text{low}}\)[8](https://arxiv.org/pdf/2412.02099?). During the mid-to-late denoising steps of this anchor pass (specifically \(t \in [0.5, 0.2]\), where semantic layout is solidified), the engine hooks into the cross-attention layers (`attn2` in SDXL's mid/up blocks; cross-attention blocks in Anima) and accumulates the spatial attention map \(M \in \mathbb{R}^{N \times M}\), where \(N = h \times w\) is the number of spatial latent tokens and \(M\) is the number of text tokens in the prompt \(y\)[8](https://arxiv.org/pdf/2412.02099?). The column vector \(M_{:,j}\) measures how strongly each spatial region attends to the \(j\)-th word token[8](https://arxiv.org/pdf/2412.02099?).

For each token \(j\), the engine converts the continuous attention map \(M\) into a binary spatial presence mask \(B \in \{0, 1\}^{N \times M}\) by comparing each spatial location's attentiveness against the spatial mean of that token across the canvas[8](https://arxiv.org/pdf/2412.02099?):

$$
B_{i,j} = \begin{cases} 1, & \text{if } M_{i,j} > \overline{M}_{:,j} \\ 0, & \text{otherwise} \end{cases}
$$

where \(\overline{M}_{:,j} = \frac{1}{N}\sum_{i=1}^N M_{i,j}\) (optionally scaled by a sensitivity multiplier \(\lambda \in [0.8, 1.2]\))[8](https://arxiv.org/pdf/2412.02099?). When the latent canvas is upscaled to high resolution and partitioned into overlapping patches \(\{P_k\}_{k=1}^K\), the binary mask \(B_{:,j}\) is bilinearly upsampled to the target grid. For a given patch \(P_k\), the token coverage ratio \(r_{k,j}\) is computed as the fraction of patch \(P_k\)'s area where \(B_{i,j} = 1\):

$$
r_{k,j} = \frac{\sum_{i \in P_k} B_{i,j}}{|P_k|}
$$

If \(r_{k,j} \ge \theta_{\text{tile}}\) (where \(\theta_{\text{tile}} \in [0.05, 0.25]\) is the **Tile Overlap Inclusion Threshold**, analogous to `sd-forge-couple`'s tile inclusion filter that prevents regions barely touching a tile from injecting their prompt)[44](https://github.com/Haoming02/sd-forge-couple), token \(j\) is retained in the patch-specific prompt \(y_k\)[3](https://lzhxmu.github.io/accdiffusion/accdiffusion.html). Global style descriptors, lighting modifiers, and camera tokens (which exhibit diffuse, low-variance attention across the entire canvas) are retained globally across all patches, while localized noun phrases (e.g., *"sun"*, *"lighthouse"*) are pruned from patches where \(r_{k,j} < \theta_{\text{tile}}\)[3](https://lzhxmu.github.io/accdiffusion/accdiffusion.html).

#### Handling Multi-Tile Objects, Subject Counts, and Dilated Window Interaction

Three critical boundary phenomena arise when objects span multiple tiles or involve plural subject counts. First, if a global or large regional prompt specifies plural entities such as *"2girls"* or *"three ships"* and spans two adjacent high-resolution tiles, passing *"2girls"* to both tiles causes each tile to synthesize two characters, yielding four characters in total[44](https://github.com/Haoming02/sd-forge-couple). Incorporating `sd-forge-couple`'s **Subject Replacement** mechanism[44](https://github.com/Haoming02/sd-forge-couple) resolves this by automatically rewriting plural quantifiers (`2girls`, `multiple trees`) into singular descriptors (`1girl`, `tree`) within sub-patches whose bounding intersection with the subject mask is less than \(65\%\) of the total subject mask area.

Second, in standard DemoFusion, dilated sampling subsamples the high-resolution latent \(z_t \in \mathbb{R}^{c \times H \times W}\) into \(D = s_h \times s_w\) interleaved sub-latents, denoising each independently without cross-window communication, which injects grainy, inconsistent global guidance[3](https://lzhxmu.github.io/accdiffusion/accdiffusion.html). Following AccDiffusion v2, the synthesis engine applies a **position-wise bijection function** \(\pi\) across the \(D\) dilated samples at each step, permuting spatial noise coordinates between adjacent dilated grids before and after denoising so that structural gradients propagate smoothly across all dilation phases[3](https://lzhxmu.github.io/accdiffusion/accdiffusion.html).

Third, global structure is determined primarily in the first \(40\%\) of denoising steps (\(t \in [1.0, 0.6]\)), whereas high-frequency micro-textures form in the final \(60\%\) (\(t \in [0.6, 0.0]\))[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf). During progressive high-resolution refinement, the skip-residual weight \(c_1(t)\) (governed by cosine decay `cosine_scale_1 = 3.0`) and bijective dilated sampling weight \(c_2(t)\) (`cosine_scale_2 = 1.0`) dominate during early steps to lock in the anchor composition, and decay toward zero in late steps so local patch prompts can synthesize crisp, unblurred textures without ghosting from the bicubic-upsampled anchor[8](https://arxiv.org/pdf/2412.02099?).

### Track 5: Regional Conditioning and True Regional LoRAs

A critical flaw in existing WebUI workflows is the conflation of three mathematically distinct operations: **region-specific text attention**, **region-specific LoRA activation**, and **genuine spatial isolation of LoRA effects**[18](https://github.com/hako-mikan/sd-webui-regional-prompter).

#### Architectural Distinction of the Three Regional Capabilities

In standard cross-attention, latent queries \(\mathbf{Q} \in \mathbb{R}^{N \times d}\) attend to a single global text context \(\mathbf{K}, \mathbf{V} \in \mathbb{R}^{M \times d}\) via \(\mathbf{R} = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}}\right)\mathbf{V}\)[45](https://arxiv.org/html/2402.05408v2). In **region-specific text attention** (`AttentionCouple` / `sd-forge-couple` / Regional Prompter `Attention` Mode), the prompt is split into \(R\) regional prompts, encoded independently into \(\{\mathbf{K}_r, \mathbf{V}_r\}_{r=1}^R\), and each cross-attention layer computes regional residuals that are multiplied by the downsampled spatial mask \(M_r \in [0, 1]^{N \times 1}\) and summed[13](https://www.sourcepulse.org/projects/2335762):

$$
\mathbf{R}_{\text{coupled}} = \sum_{r=1}^R \overline{M}_r \odot \left(\text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}_r^T}{\sqrt{d}}\right)\mathbf{V}_r\right)
$$

where \(\overline{M}_r\) is normalized across overlapping regions such that \(\sum_{r=1}^R \overline{M}_{r}(i) = 1\) at every pixel \(i\)[22](https://github.com/pamparamm/ComfyUI-ppm/issues/21). This executes in a *single* U-Net/DiT forward pass and cleanly separates text concepts, preventing color bleed[13](https://www.sourcepulse.org/projects/2335762).

However, attention coupling fails to achieve **regional LoRA isolation**[18](https://github.com/hako-mikan/sd-webui-regional-prompter). A Low-Rank Adaptation (LoRA) adapter does not merely add tokens to \(\mathbf{K}_r, \mathbf{V}_r\); it patches weight matrices \(W' = W + \alpha \Delta W = W + \alpha (B A)\) across **both** the text encoder (`CLIP` / `Qwen`) and **all** self-attention (`attn1`), cross-attention (`attn2`), and feed-forward (`ff` / MLP) projections inside the U-Net or DiT[18](https://github.com/hako-mikan/sd-webui-regional-prompter). If two character or style LoRAs (`LoRA_A` and `LoRA_B`) are loaded simultaneously in `Attention` mode, their weight deltas \(\Delta W_A + \Delta W_B\) are merged globally into the model backbone[18](https://github.com/hako-mikan/sd-webui-regional-prompter). Even if their trigger words are isolated into separate regional prompts, `attn1` (self-attention) and `ff` layers apply both LoRAs across 100% of the canvas, causing severe identity mixing and style contamination[21](https://github.com/hako-mikan/sd-webui-regional-prompter/discussions/272).

To separate LoRAs, `sd-webui-regional-prompter` introduced `Latent` calculation mode, which runs the entire diffusion model \(R\) separate times per step—activating only `LoRA_r` and `Prompt_r` on the \(r\)-th forward pass—and blends the predicted noise tensors \(\epsilon_r\) using the spatial masks[18](https://github.com/hako-mikan/sd-webui-regional-prompter). While this isolates LoRA weights, it scales generation time linearly by \(R \times\) (e.g., 5 regions = \(5\times\) slower)[18](https://github.com/hako-mikan/sd-webui-regional-prompter), frequently suffers from non-deterministic state restoration bugs and weakened LoRA intensity when switching weights mid-step[47](https://github.com/hako-mikan/sd-webui-regional-prompter/issues/400), and still allows visual leakage across mask boundaries due to global self-attention receptive fields during each full-canvas pass[21](https://github.com/hako-mikan/sd-webui-regional-prompter/discussions/272).

#### Lessons from Multi-Instance Generation Research (MIGC and MIGC++)

Academic Multi-Instance Generation research—specifically **MIGC (CVPR 2024)** and **MIGC++ (2024–2025)**—provides two vital architectural mechanisms to eliminate cross-region attribute leakage without running \(R\) full model passes[20](https://arxiv.org/pdf/2407.02329?). First, MIGC++ demonstrates that encoding multiple instance descriptions together in a single CLIP/LLM pass causes contextual embedding bleed before diffusion even begins[20](https://arxiv.org/pdf/2407.02329?). Consequently, each regional prompt—and any text-encoder LoRA associated with that region—must be encoded in strict isolation during the text-encoding phase, caching separate conditioning tensors \(\mathbf{C}_r\) per region[6](https://github.com/GumGum10/forge-anima-3.8B). Second, MIGC++ replaces standard cross-attention in the mid-block and deep up-blocks with parallel **Instance Shaders** (using Enhancement Attention masked to each instance's 2D position map) and **training-free Refined Shaders** that prevent background and neighboring instances from absorbing instance-specific visual attributes, followed by a **Consistent-MIG** masked replacement step that preserves non-modified background regions[20](https://arxiv.org/pdf/2407.02329?).

#### Ranked Implementable Strategies for SDXL and Anima in Forge Neo

To provide both fast workflow responsiveness and strict LoRA isolation, the extension implements a **Three-Tier Regional Conditioning & LoRA Engine** utilizing Forge Neo's native `on-the-fly LoRA` patching (`ModelPatcher`)[2](https://github.com/Haoming02/sd-webui-forge-classic):

| Rank & Strategy Name | Mechanism & Execution Overhead | LoRA Isolation Fidelity | SDXL (U-Net) Implementation | Anima (2B / 3.8B DiT) Implementation |
| --- | --- | --- | --- | --- |
| **Tier 1 (Default): Masked On-the-Fly LoRA Hook + Refined Attention Couple** | **1.05x–1.15x Latency Overhead (Single Forward Pass)**. Text encoders run sequentially per region with only that region's TE-LoRA active, caching \(\mathbf{C}_r\)[6](https://github.com/GumGum10/forge-anima-3.8B). During the single U-Net/DiT forward pass, linear/conv layers wrapped by `ModelPatcher` evaluate the base output \(y_0 = Wx\) once, and apply regional LoRA deltas masked in-place by downsampled spatial mask \(M_r\): \(y = y_0 + \sum_{r \in \mathcal{L}} M_r \odot (\alpha_r B_r A_r x)\). Cross-attention uses MIGC++-style masked `AttentionCouple`[2](https://github.com/Haoming02/sd-webui-forge-classic). | **High (90–95% Isolation)**. Completely isolates feed-forward, `attn2`, and `q/k/v` projection LoRA deltas to their exact pixel masks \(M_r\) within a single pass. Minor leakage can occur only via `attn1` (self-attention) keys crossing region borders. | Hooks `Linear` and `Conv2d` LoRA patches inside `UnetPatcher` (`KModel`), dynamically reshaping the 2D mask \(M_r\) to match the current U-Net block's feature map resolution (\(128^2, 64^2, 32^2\))[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf). | Hooks DiT `Linear` layers (`qkv`, `proj`, `mlp`) via `ModelPatcher`. Because DiT tokens are flattened patch sequences \((B, L, D)\), the 2D mask \(M_r\) is flattened to \((1, L, 1)\) and broadcast-multiplied with \(\Delta W_r x\)[10](https://note.com/198619891990/n/nf8d0cb57d57e). |
| **Tier 2 (Strict): Self-Attention Block-Diagonal Masking + Tier 1** | **1.15x–1.25x Latency Overhead (Single Forward Pass)**. Combines Tier 1's spatially masked \(\Delta W_r\) with an attention bias mask \(A_{\text{bias}} \in \{0, -\infty\}^{L \times L}\) injected into `attn1` (self-attention) during early steps (\(t \in [1.0, 0.7]\)) so queries inside Region \(A\) cannot attend to keys inside conflicting LoRA Region \(B\), while both can attend to the shared background region. | **Strict (99% Isolation)**. Eliminates self-attention style/identity bleed between conflicting character or art-style LoRAs while maintaining coherent lighting with the global background[20](https://arxiv.org/pdf/2407.02329?). | Passed as `attn_bias` to PyTorch `scaled_dot_product_attention` (SDPA) or `xformers` in `DownBlock1/2` and `UpBlock1/2` during steps \(0 \dots 0.3T\)[2](https://github.com/Haoming02/sd-webui-forge-classic). | Passed as block-sparse/masked attention mask in Anima's DiT self-attention blocks for the first \(30\%\) of rectified flow / diffusion steps[2](https://github.com/Haoming02/sd-webui-forge-classic). |
| **Tier 3 (Fallback): Tile-Aligned Regional Sub-Batching** | **1.0x Overhead during Tiled High-Res Passes; \(K\times\) at Base Res**. During tiled high-resolution refinement, each tile \(P_k\) only activates the LoRAs whose masks intersect \(P_k\) (\(r_{k,\text{lora}} \ge \theta_{\text{tile}}\))[44](https://github.com/Haoming02/sd-forge-couple). | **Absolute (100% Hardware Isolation across non-overlapping tiles)**. LoRAs absent from a tile are zeroed out in `ModelPatcher` for that tile's forward pass. | Native integration with Forge Neo's `on-the-fly LoRA` strength multiplier per tile batch[2](https://github.com/Haoming02/sd-webui-forge-classic). | Native integration with Anima's `ModelPatcher` and `Semantic Connector v2` per-tile batch[2](https://github.com/Haoming02/sd-webui-forge-classic). |

### Track 8: Additional Structural Guidance

High-resolution regional synthesis frequently benefits from optional structural conditioning (pose, depth, lineart, or color-coded region masks) without forcing the system to keep a 2.5 GB full ControlNet model resident in VRAM[49](https://civitai.com/models/2604674/controlnet-lllite-anima-base-v1).

#### Native Integration with Kohya's ControlNet-LLLite and Anima Region ControlNet

Inspection of the Forge Neo repository (`Haoming02/sd-webui-forge-classic`, `neo` branch) confirms two crucial built-in capabilities recently added to the core engine[2](https://github.com/Haoming02/sd-webui-forge-classic). Unlike standard ControlNet (which duplicates the entire U-Net encoder and consumes 1.5–2.5 GB VRAM), Kohya's `ControlNet-LLLite` injects lightweight LoRA-like conditioning modules directly into the base model, requiring only **7.48 MB** for 8-bit Anima Base v1 checkpoints (e.g., `AB1_LINEART`) and ~15–50 MB for SDXL checkpoints (`kohya_controllllite_xl_canny`, `kohya_controllllite_xl_depth_anime`)[49](https://civitai.com/models/2604674/controlnet-lllite-anima-base-v1). Crucially, Forge Neo has already updated `LLLite ControlNet` for both SDXL and Anima to support **MultiDiffusion (tiled slicing)** natively[2](https://github.com/Haoming02/sd-webui-forge-classic). When the synthesis engine streams a latent tile \(P_k\) at coordinates \([x_0, y_0, x_1, y_1]\), Forge Neo's LLLite patcher automatically slices the corresponding pixel crop from the conditioning image.

In addition, Forge Neo natively supports Kohya's `Region ControlNet` for Anima (`controlnet end percent step` recommended at `0.45`), which accepts a raster control image consisting of a white background with solid colored regions and binds spatial structural boundaries directly into Anima's DiT blocks[2](https://github.com/Haoming02/sd-webui-forge-classic). Currently, users must manually paint this color mask in external image editors and upload it to the ControlNet unit[52](https://www.reddit.com/r/StableDiffusion/comments/1u3q7mn/regional_controlnet_for_anima/). The extension eliminates this manual step by **automatically rasterizing the Multilayer Regional Canvas or CRF JSON regions into an Anima Region ControlNet color map in memory**, feeding it directly into the `LLLite` Region ControlNet unit for steps \(t \in [1.0, 0.55]\) (`end_percent = 0.45`) without manual file handling[2](https://github.com/Haoming02/sd-webui-forge-classic).

#### Zero-VRAM Structural Guidance from Intermediate Anchor Compositions

To provide depth, canny edge, or sketch guidance at \(4096 \times 4096\) without loading heavy external preprocessors while the diffusion model is on the GPU, the engine leverages the progressive pipeline's Stage 1 (\(1024 \times 1024\)) anchor image. After Stage 1 completes, the \(1024 \times 1024\) latent is decoded via `TAESD` (tiny autoencoder, `<15 MB` VRAM, natively supported in Forge Neo[2](https://github.com/Haoming02/sd-webui-forge-classic)) or standard VAE into an RGB image. A pure PyTorch tensor Kornia/Sobel-Canny filter (0 MB model weights, `<2 ms` execution time) extracts crisp structural edges from the upscaled anchor image, or optionally runs a lightweight preprocessor (`Depth_leres` / `Anime-Face-Segmentation`[50](https://stable-diffusion-art.com/controlnet-sdxl/)) while the main diffusion model is temporarily offloaded between Stage 1 and Stage 2, immediately unloading the preprocessor before Stage 2 tiled diffusion begins.

Furthermore, combining MIGC++'s **Consistent-MIG** algorithm[20](https://arxiv.org/pdf/2407.02329?) with Forge Neo's full-precision mask blur blending (`Settings/img2img`)[2](https://github.com/Haoming02/sd-webui-forge-classic) enables localized region-aware inpainting and regeneration. Users can lock four out of five regions on a \(4096 \times 4096\) canvas and regenerate only the second region; the engine caches the final latent \(z_0\) in CPU RAM, extracts only the tiles overlapping the modified region, blends noise using the feather-blurred mask at `fp32` precision to prevent quantization banding[2](https://github.com/Haoming02/sd-webui-forge-classic), and re-denoises only those tiles in seconds.

## Deliverable 2: Forge Neo Internals, Model Compatibility, and Low-VRAM Engineering (FEASIBILITY.md)

### Track 2: Forge Neo Architecture and SDXL vs. Anima Model Internals

Designing a stable extension requires targeting the exact internal abstractions of Forge Neo (`Haoming02/sd-webui-forge-classic`, `neo` branch, currently at v2.28–v2.30 running on Gradio 4.40.0, Python 3.11–3.13, and PyTorch 2.10+cu130)[2](https://github.com/Haoming02/sd-webui-forge-classic).

#### Forge Neo Backend Hooks and `ModelPatcher` Lifecycle

Unlike legacy AUTOMATIC1111 (which mutated global PyTorch modules directly), Forge and Forge Neo rewrite the backend around ComfyUI's declarative patching architecture (`backend/memory_management.py`, `backend/loader.py`, `ModelPatcher`, and `UnetPatcher` wrapping `KModel`)[2](https://github.com/Haoming02/sd-webui-forge-classic). Rather than monkey-patching `forward()` methods—which breaks when Forge offloads layers or switches models—extensions must register patches inside `p.sd_model.forge_objects.unet.model_options` (the `UnetPatcher` / `ModelPatcher` instance cloned per generation run)[10](https://note.com/198619891990/n/nf8d0cb57d57e). Specifically, `set_model_attn1_patch` and `set_model_attn1_replace` intercept self-attention inputs and outputs, serving as the injection point for HiDiffusion MSW-MSA, Anima AR-RoPE tiled attention, and Tier 2 LoRA block-diagonal masking[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf). Cross-attention is governed by `set_model_attn2_patch` and `set_model_attn2_replace`, which are utilized by `sd-forge-couple` for regional text conditioning and by the Track 4 subsystem for accumulating low-resolution token attention maps \(M \in \mathbb{R}^{N \times M}\)[8](https://arxiv.org/pdf/2412.02099?). At the outer loop level, `set_model_unet_function_wrapper` wraps the entire denoising step call `apply_model(x, timestep, **c)`[58](https://github.com/Burgstall-labs/ComfyUI-Seamless-Equirectangular), acting as the master hook where the progressive/tiled scheduler intercepts the full-canvas latent, slices or streams GPU tiles, adjusts per-tile conditioning and RoPE coordinates, and reassembles predictions[58](https://github.com/Burgstall-labs/ComfyUI-Seamless-Equirectangular). Finally, `transformer_options["patches_replace"]` permits block-specific structural overrides, such as patching `DownBlock1` and `UpBlock1` in SDXL for HiDiffusion RAD/RAU[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf).

Forge Neo automatically selects the first available attention backend in a strict priority order: `SageAttention` \(\rightarrow\) `FlashAttention` \(\rightarrow\) `ComfyKitchenAttention` \(\rightarrow\) `xformers` (not supported on RTX 50-series) \(\rightarrow\) `PyTorch SDPA` (`scaled_dot_product_attention`) \(\rightarrow\) `Basic`[2](https://github.com/Haoming02/sd-webui-forge-classic), alongside optional experimental accelerators (`Radial Attention` via `SpargeAttn` and `Sol-Attn`)[2](https://github.com/Haoming02/sd-webui-forge-classic). A critical compatibility constraint is that `SageAttention` uses quantized INT8/FP8 attention kernels that require *both* positive and negative prompts to be present (throwing an error if unconditional batching is omitted)[59](https://www.stablediffusiontutorials.com/2025/11/forge-neo-installation.html) and does not support arbitrary dense float `attn_bias` masks. Consequently, when the extension uses `set_model_attn2_replace` (which computes regional attention per mask and sums the output profiles, as in `sd-forge-couple`), `SageAttention` works natively because each regional call is a standard unmasked attention pass multiplied post-softmax by the spatial mask \(\overline{M}_r\). When Tier 2 intra-pass `attn1` bias masking is enabled, the hook dynamically dispatches that specific layer call to `PyTorch SDPA` (`torch.nn.functional.scaled_dot_product_attention`), which supports arbitrary boolean and float `attn_mask` tensors natively across all GPU architectures[2](https://github.com/Haoming02/sd-webui-forge-classic).

[Interactive visualization 2 (cross_origin_reference_only)](#)

#### Internal Representation of SDXL vs. Anima (2B Base and 3.8B v2)

The table below contrasts the internal representations of SDXL and the Anima family as implemented in Forge Neo and the `forge-anima-3.8B` extension (`sd-forge-anima-3-8b`)[6](https://github.com/GumGum10/forge-anima-3.8B):

| Architectural Dimension | SDXL 1.0 / Turbo / DMD | Anima Base v1 (~2B) | Anima 3.8B (v1 & v2 Bundle) |
| --- | --- | --- | --- |
| **Backbone Topology** | Convolutional U-Net (`DownBlocks`, `MidBlock`, `UpBlocks`) with spatial 2D ResNet + Transformer blocks[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf). | ~2B parameter Diffusion Transformer (DiT) built on NVIDIA Cosmos (`circlestone-labs/Anima`)[5](https://huggingface.co/circlestone-labs/Anima/discussions/35). | 3.8B expanded DiT (`Anima-3.8B-v2.safetensors`) with progressive cross-adapter[6](https://github.com/GumGum10/forge-anima-3.8B). |
| **Precision & Dtype** | `fp16` or `bf16` weights (`~6.5 GB` in `fp16`). | Strict `bf16` native training and inference (`~4.0 GB` in `bf16`; cannot be trained in `fp16`)[5](https://huggingface.co/circlestone-labs/Anima/discussions/35). | `bf16` native or quantized (`Q4 GGUF` / `W6A8` supported in Forge Neo 2.29.2+)[5](https://huggingface.co/circlestone-labs/Anima/discussions/35). |
| **Text Conditioning Path** | Dual CLIP (`CLIP-ViT-L/14` + `OpenCLIP-ViT-bigG/14`), concatenated along channel dim (`2048-d`) plus pooled `adm` vector (`1280-d`). | `qwen_3_06b_base.safetensors` (`Qwen3 0.6B Base`) projected via an internal LLM Adapter[5](https://huggingface.co/circlestone-labs/Anima/discussions/35). | **Dual Sequential Path**: Native `qwen_3_06b_base` + `qwen35_4b.safetensors` (`4.8 GB` Qwen3.5 4B) linked via **Semantic Connector v2** (`strength = 1.0`)[6](https://github.com/GumGum10/forge-anima-3.8B). |
| **Timestep Conditioning** | Sinusoidal timestep embedding added to ResNet blocks; static text conditioning \(\mathbf{C}\) across all steps. | Flow-matching / DiT timestep modulation (`AdaLN-Single` / `AdaLN-Zero`). | **Timestep-Aware Extraction**: `Semantic Connector v2` remains active during *every* denoising step because its feature extraction from Qwen3.5 4B outputs is timestep-dependent[6](https://github.com/GumGum10/forge-anima-3.8B). |
| **Negative Prompt Path** | Standard batch concatenation `[uncond, cond]` through dual CLIP. | Standard batch concatenation through `qwen_3_06b_base` + LLM Adapter. | Asymmetric by default: Positive prompt uses `Qwen3.5 4B + Semantic Connector v2`; negative prompt stays on native `qwen_3_06b_base` unless *"Use adapter on negative prompt"* is toggled[6](https://github.com/GumGum10/forge-anima-3.8B). |
| **Positional Encoding** | Implicit via zero-padded 2D convolutions + `adm` micro-conditioning `(original_size, crops_coords_top_left, target_size)`. | **3D RoPE** (`rope_scale = [1, 4, 4]` for `[T, H, W]` where \(T=1\) for images)[15](https://ostris.com/docs/ai-toolkit/models/anima). | **3D RoPE** (`rope_scale = [1, 4, 4]`) applied to flattened patch tokens \((B, L, D)\)[15](https://ostris.com/docs/ai-toolkit/models/anima). |
| **VAE Architecture** | `sdxl_vae.safetensors` (\(8\times\) spatial compression, 4 latent channels). | `qwen_image_vae.safetensors` (Qwen-Image VAE, shared with Krea 2 and Qwen-Image 2.1)[6](https://github.com/GumGum10/forge-anima-3.8B). | `qwen_image_vae.safetensors` (exhibits known grid artifacts unless filtered via Forge Neo's **DeGrid** or circular/tiled padding)[2](https://github.com/Haoming02/sd-webui-forge-classic). |

#### Preserving Anima's 3D RoPE Positional Information During Tiled and High-Resolution Processing

A critical finding of Track 2 is how tiled processing interacts with Anima's **3D Rotary Position Embeddings (`rope_scale = [1, 4, 4]`)**[15](https://ostris.com/docs/ai-toolkit/models/anima) compared to SDXL's convolutions. When a standard MultiDiffusion script crops a \(128 \times 128\) latent tile \(P_k\) from a \(512 \times 512\) high-resolution latent canvas and passes \(P_k\) into Anima, the DiT's patchifier generates local token coordinates \((y_{\text{local}}, x_{\text{local}}) \in [0, H_{\text{tile}}-1] \times [0, W_{\text{tile}}-1]\) starting at \((0, 0)\) for *every* tile. Because Anima relies on 3D RoPE to determine where a patch sits within the global composition (e.g., top-left sky vs. bottom-right foreground)[62](https://mscvprojects.ri.cmu.edu/2026teamf8/our_contribution/), resetting RoPE coordinates to \((0,0)\) per tile destroys global spatial awareness and causes severe structural repetition and horizon shifts across tiles[16](https://arxiv.org/html/2508.15774v2).

To resolve this across both model families, the engine applies architecture-specific coordinate offsetting. For SDXL, when denoising a tile at pixel offset \((y_0, x_0)\) on a canvas of size \((H_{\text{full}}, W_{\text{full}})\), the engine dynamically updates SDXL's `adm` micro-conditioning tuple `crops_coords_top_left = (y_0, x_0)` and `original_size = (H_full, W_full)`. For Anima, the engine hooks Anima's `pe_embedder` / RoPE generator inside `ModelPatcher` (analogous to `FiL Krea 2 Tiled Diffusion` and `ComfyUI-Anima` bbox RoPE placement nodes[17](https://comfy.icu/node/FiLKrea2TiledDiffusion)). When evaluating tile \(P_k\) with latent top-left offset \((\ell_y^{(k)}, \ell_x^{(k)})\), the RoPE coordinate grid is generated using the **global canvas coordinates** normalized into the native pretrained RoPE frequency range via **Time-Aware / Adaptively Rectified RoPE (AR-RoPE)**[16](https://arxiv.org/html/2508.15774v2):

$$
\tilde{y}(i) = \ell_y^{(k)} \cdot \frac{H_{\text{base}}}{H_{\text{full}}} + i \cdot s_{\text{rope}}(t), \quad \tilde{x}(j) = \ell_x^{(k)} \cdot \frac{W_{\text{base}}}{W_{\text{full}}} + j \cdot s_{\text{rope}}(t)
$$

where \(s_{\text{rope}}(t)\) interpolates from \(\frac{L_{\text{base}}}{L_{\text{full}}}\) at \(t=1.0\) (preserving global relative wavelength across the canvas so the DiT recognizes the tile's exact global position) toward \(1.0\) as \(t \rightarrow 0\) (restoring local patch-to-patch high-frequency spacing so textures remain sharp)[16](https://arxiv.org/html/2508.15774v2). Furthermore, because `Semantic Connector v2` in Anima 3.8B extracts timestep-aware features from the cached `Qwen3.5 4B` hidden states on every step[6](https://github.com/GumGum10/forge-anima-3.8B), the `ModelPatcher` wrapper caches the raw `Qwen3.5 4B` hidden state tensor \(\mathbf{H}_{\text{qwen}}^{(r)}\) in CPU/GPU memory once per region \(r\) before sampling begins, and invokes only the lightweight `Semantic Connector v2` projection module at each timestep \(t\) per tile[6](https://github.com/GumGum10/forge-anima-3.8B).

#### Accelerated Checkpoints (DMD, Turbo, Rectified Flow, Krea 2 Turbo)

Accelerated checkpoints (4-to-8 step SDXL-Turbo, DMD2, Lightning, and 8-step Rectified Flow / Krea 2 Turbo / Anima accelerated merges) compress structural formation into the first 1–2 steps[2](https://github.com/Haoming02/sd-webui-forge-classic). In 50-step DDIM/DPM++, HiDiffusion's RAU-Net switches off at step \(T_1 = 20\) (\(40\%\) of steps)[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf). For a 4-step Turbo or 8-step DMD/Rectified model, hardcoding step counts fails; therefore, all switching thresholds (RAU-Net \(T_1\), Region ControlNet `end_percent = 0.45`[52](https://www.reddit.com/r/StableDiffusion/comments/1u3q7mn/regional_controlnet_for_anima/), and Tier 2 LoRA masking) are parameterized strictly as **normalized progress fractions** \(\tau \in [0.0, 1.0]\) (e.g., active for steps where \(\text{step} / \text{total\_steps} \le 0.40\), which equals steps 1–2 in a 4-step run and steps 1–3 in an 8-step run)[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf). Additionally, while 30-step models use full noise inversion (\(t = 1.0 \rightarrow 0.0\)) in DemoFusion's progressive stages, 4–8 step DMD/Turbo models suffer from burnt contrast or structural drift under full re-noising. For accelerated checkpoints, Stage 2+ progressive passes operate at reduced `denoise` strength (\(0.25\text{–}0.45\), matching `FiL Krea 2 Tiled Diffusion`'s optimal `8 steps, denoise 0.22–0.25`[17](https://comfy.icu/node/FiLKrea2TiledDiffusion)) or use **RectifiedHR Noise Refresh** at the final 2 steps[35](https://arxiv.org/html/2503.02537v3), paired with Forge Neo's built-in `RescaleCFG` (for `v_pred` / `ztsnr` models) and `MaHiRo` CFG scaling[2](https://github.com/Haoming02/sd-webui-forge-classic).

### Track 3: Low-VRAM Engineering and Memory Budget Profiling

Generating a \(4096 \times 4096\) image involves a latent tensor of shape \([1, C, 512, 512]\) (\(262,144\) spatial positions, or \(65,536\) patch tokens in a \(2\times 2\) patchified DiT). Keeping the full computational graph, intermediate activations, text encoders, and VAE decoder on the GPU simultaneously requires \(>28\text{ GB}\) of VRAM. Through aggressive low-VRAM engineering aligned with Forge Neo's `memory_management.py`[2](https://github.com/Haoming02/sd-webui-forge-classic), \(4096 \times 4096\) synthesis can be executed reliably on **6 GB to 8 GB GPUs**, and \(2048 \times 3072\) on **4 GB GPUs**.

#### Core Low-VRAM Systems Mechanisms

To decouple canvas resolution from GPU memory limits, the master high-resolution latent canvas \(\mathbf{Z}_t \in \mathbb{R}^{1 \times C \times H \times W}\), along with the accumulation buffer \(\mathbf{Z}_{\text{accum}}\) and weight normalization buffer \(\mathbf{W}_{\text{accum}}\), are allocated in **page-locked (pinned) CPU system RAM** (`torch.empty(..., pin_memory=True, dtype=torch.float16)`). During each denoising step, only a micro-batch of tiles (`view_batch_size = 1` on 4–6 GB VRAM; `2–4` on 8–12 GB VRAM[12](https://github.com/PRIS-CV/DemoFusion)) of size \(128 \times 128\) latent pixels (\(1024 \times 1024\) image pixels) is transferred asynchronously to the GPU over a dedicated CUDA stream (`tensor.to(device, non_blocking=True)`). While the GPU executes `apply_model()` on Tile \(k\), the CUDA copy engine prefetches Tile \(k+1\) from pinned RAM and writes Tile \(k-1\)'s denoised prediction back to \(\mathbf{Z}_{\text{accum}}\) on the CPU. Peak GPU activation memory during a \(4096 \times 4096\) or \(8192 \times 8192\) run thus remains strictly identical to a single \(1024 \times 1024\) generation.

In Anima 3.8B v2, `qwen_3_06b_base` (~1.2 GB) and `qwen35_4b.safetensors` (`4.8 GB` in bf16, or `~2.6 GB` as Q4 GGUF)[6](https://github.com/GumGum10/forge-anima-3.8B) cannot coexist in VRAM alongside the 3.8B DiT (`~7.6 GB` in bf16) on 8–12 GB cards. Following `forge-anima-3.8B`'s sequential execution architecture[6](https://github.com/GumGum10/forge-anima-3.8B), the extension encodes all \(R\) regional prompts **sequentially before loading the diffusion backbone**. First, `qwen_3_06b_base` is loaded to iterate through Regions \(1 \dots R\), applying each region's TE-LoRA on-the-fly, moving outputs to CPU pinned RAM, and unloading `qwen_3_06b_base`[6](https://github.com/GumGum10/forge-anima-3.8B). Next, `qwen35_4b` (bf16 or Q4 GGUF) is loaded to process Regions \(1 \dots R\), moving hidden state tensors to CPU pinned RAM before unloading `qwen35_4b` and invoking `torch.cuda.empty_cache()`[5](https://huggingface.co/circlestone-labs/Anima/discussions/35). An SHA-256 hash of `(prompt_r, lora_hashes_r, model_hash)` is computed for each region so that if a user modifies only Region 2's prompt or mask on a subsequent run, Regions 1, 3, 4, and 5 are served instantaneously from the CPU conditioning cache (`0 ms` re-encoding)[5](https://huggingface.co/circlestone-labs/Anima/discussions/35).

On 4 GB and 6 GB GPUs where the full SDXL (`6.5 GB`) or Anima 3.8B (`7.6 GB`) backbone exceeds available VRAM, Forge Neo's `memory_management.py` automatically streams model layers per forward pass[2](https://github.com/Haoming02/sd-webui-forge-classic). However, if `view_batch_size = 1` is used across 16 tiles, a naive loop would stream model weights across PCIe \(16 \times\) per step. To prevent PCIe bus thrashing under low-VRAM layer-streaming modes, the engine supports `Q4 GGUF` / `W6A8` / `NF4` quantized backbones (supported natively in Forge Neo 2.29.2+[55](https://note.com/hirorohi03/n/nb5194b250730?hl=en)) so the entire 2B/3.8B DiT (`~1.8–2.8 GB` quantized) remains 100% resident in VRAM even on a 6 GB GPU.

At the end of the diffusion loop, decoding a \(4096 \times 4096\) latent in a single VAE pass would require \(>18\text{ GB}\) VRAM[12](https://github.com/PRIS-CV/DemoFusion). The extension therefore enforces Forge Neo's **Tiled VAE** (`multi_decoder = True`[12](https://github.com/PRIS-CV/DemoFusion)) paired with Forge Neo's built-in **tiled `Conv2d` for VAE** and GPU tile composition[2](https://github.com/Haoming02/sd-webui-forge-classic). Furthermore, because `qwen_image_vae.safetensors` (used by Anima, Krea 2, and Qwen-Image 2.1) is known to produce subtle grid-like boundary artifacts during high-resolution tiled decoding[55](https://note.com/hirorohi03/n/nb5194b250730?hl=en), the extension automatically pipes the decoded image through Forge Neo 2.30's built-in **DeGrid** filter[2](https://github.com/Haoming02/sd-webui-forge-classic) and optional circular convolution padding (`EquirectCircularVAE` style[58](https://github.com/Burgstall-labs/ComfyUI-Seamless-Equirectangular)) to ensure seamless 4K outputs. Finally, in multi-stage progressive generation (e.g., Stage 1: \(1024^2 \rightarrow\) Stage 2: \(2048^2 \rightarrow\) Stage 3: \(4096^2\)), completion of each stage immediately writes the CPU latent tensor and a `TAESD` preview PNG to a temporary recovery buffer on disk[2](https://github.com/Haoming02/sd-webui-forge-classic). If an `OutOfMemoryError` or user interruption occurs during Stage 3, the engine catches the exception, clears the CUDA cache, restores the `ModelPatcher` state, returns the completed Stage 2 (\(2048^2\)) image, and automatically steps down `view_batch_size` (e.g., \(4 \rightarrow 2 \rightarrow 1\)) and attention chunk size before retrying Stage 3.

#### Quantitative Memory Budget and Resolution Feasibility Matrix

Based on model footprints, tile activation sizes (\(128 \times 128\) latent tiles), and tiled VAE decoding, the practical limits, execution times (assuming 25 steps on a midrange GPU per tier), and required configurations are established below:

| VRAM Budget Tier | Target Model Configuration | Max Practical Native Resolution | Recommended Mode & `view_batch_size` | Peak VRAM / System RAM Usage | Estimated Execution Cost vs. \(1024^2\) Baseline | Quality & Operational Trade-offs |
| --- | --- | --- | --- | --- | --- | --- |
| **4 GB VRAM** *(e.g., GTX 1650 / RTX 3050 Laptop)* | **Anima 2B** (bf16 streamed or Q4/W6A8) or **SDXL** (streamed via `lowvram` + `--cuda-malloc`)[55](https://note.com/hirorohi03/n/nb5194b250730?hl=en). | **\(2048 \times 2048\)** *(Hard ceiling: \(2560 \times 2560\))* | Progressive Tiled (`view_batch_size = 1`), Tiled VAE + Tiled `Conv2d`[2](https://github.com/Haoming02/sd-webui-forge-classic). | `3.6–3.9 GB` VRAM / `12–16 GB` System RAM | **~6.5x–9x** (`~90–140s` for \(2048^2\)) | HiDiffusion direct \(2048^2\) OOMs; must use \(1024^2\) tiled streaming. Quantized text encoder (Q4 GGUF) mandatory to avoid swap thrashing[6](https://github.com/GumGum10/forge-anima-3.8B). |
| **6 GB VRAM** *(e.g., GTX 1060 6GB / RTX 2060 / 3060 Laptop)* | **Anima 2B** (resident bf16) / **Anima 3.8B** (Q4/W6A8) / **SDXL** (`medvram-sdxl`)[50](https://stable-diffusion-art.com/controlnet-sdxl/). | **\(3072 \times 3072\)** *(Practical sweet spot: \(2048 \times 3072\))* | Progressive Tiled (`view_batch_size = 1`), Sage/SDPA Attention, Tiled VAE[2](https://github.com/Haoming02/sd-webui-forge-classic). | `5.4–5.8 GB` VRAM / `16 GB` System RAM | **~4.5x** at \(2048^2\) (`~45s`); **~11x** at \(3072^2\) (`~120s`) | Anima 2B fits in VRAM during tile loop after Qwen TE offload[6](https://github.com/GumGum10/forge-anima-3.8B), making Anima 2B \(2\times\) faster than unquantized SDXL on 6 GB cards. |
| **8 GB VRAM** *(e.g., RTX 3070 / 4060 / 4060 Ti 8GB)* | **SDXL** (fp16 resident) / **Anima 2B** (bf16 resident) / **Anima 3.8B v2** (Q4/W8 or streamed bf16)[6](https://github.com/GumGum10/forge-anima-3.8B). | **\(4096 \times 4096\)** *(Direct HiDiffusion up to \(2048 \times 2048\))* | **SDXL \(2048^2\)**: Direct HiDiffusion (`1.5–2.6x` speedup)[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf). **\(4096^2\)**: Progressive Tiled (`view_batch_size = 2`). | `7.2–7.8 GB` VRAM / `16–24 GB` System RAM | **\(2048^2\) (HiDiff)**: `~12–18s`; **\(4096^2\) (Tiled)**: `~110–160s` (`~10x`) | Sweet spot for consumer hardware. Tier 1 Masked LoRA deltas run with zero disk/RAM weight swapping for SDXL and Anima 2B. |
| **12 GB VRAM** *(e.g., RTX 3060 12GB / 4070 / 4070 Super)* | **SDXL** (fp16) / **Anima 3.8B v2** (bf16 DiT resident; Qwen3.5 4B sequentially offloaded)[6](https://github.com/GumGum10/forge-anima-3.8B). | **\(4096 \times 4096\)** *(Ceiling: \(6144 \times 4096\))* | **Up to \(2560^2\)**: Direct HiDiffusion / AR-RoPE. **\(4096^2\)**: Progressive Tiled (`view_batch_size = 4`). | `10.5–11.6 GB` VRAM / `24–32 GB` System RAM | **\(2048^2\)**: `~9–14s`; **\(4096^2\)**: `~65–95s` (`~7x`) | Full `Anima-3.8B-v2` + `Semantic Connector v2` stays resident in VRAM during the entire tile denoising loop once `qwen35_4b` is offloaded to RAM[6](https://github.com/GumGum10/forge-anima-3.8B). |
| **24 GB VRAM (Ref)** *(e.g., RTX 3090 / 4090 / A6000)* | All SDXL & Anima 3.8B v2 models + LLLite ControlNet resident in `bf16`[12](https://github.com/PRIS-CV/DemoFusion). | **\(8192 \times 8192\)** (via FreeScale + Tiled Streamer)[9](https://arxiv.org/html/2412.09626v1) | **Up to \(3072^2\)**: Direct / Hybrid. **\(4096^2–8192^2\)**: Progressive (`view_batch_size = 8–16`)[12](https://github.com/PRIS-CV/DemoFusion). | `18–23.2 GB` VRAM / `32–64 GB` System RAM | **\(4096^2\)**: `~35–50s`; **\(8192^2\)**: `~180–240s` | Tiled VAE decoder (`multi_decoder = True`) remains mandatory above \(3072 \times 3072\) even on 24 GB GPUs[12](https://github.com/PRIS-CV/DemoFusion). |

### Track 9: Ecosystem Compatibility, Stability, and Legal Code Reuse

#### Extension Ecosystem Compatibility Matrix and Failure Recovery

Operating inside Forge Neo requires coexisting cleanly with popular extensions and built-in scripts (`sd-forge-couple`, `ControlNet` / `LLLite`, `ADetailer`, `sd-dynamic-prompts`, `NegPiP`, and `Regional Prompter`)[2](https://github.com/Haoming02/sd-webui-forge-classic).

| Extension / Subsystem | Interaction Vector & Potential Collision | Verified Resolution & Stability Protocol |
| --- | --- | --- |
| **`sd-forge-couple`**[13](https://www.sourcepulse.org/projects/2335762) | Shares `set_model_attn2_replace` hook and parses prompt lines/separators[19](https://comfy.icu/node/AttentionCouple). If both extensions are enabled simultaneously, `attn2` replacements overwrite each other. | **Auto-Detect & Supersede**: Detect if `sd-forge-couple` is active in `p.scripts`. Offer one-click import of `sd-forge-couple` Advanced/Mask JSON mappings into the Multilayer Canvas, and automatically disable `sd-forge-couple`'s hook when the extension is active to prevent double-patching. |
| **`sd-dynamic-prompts`**[44](https://github.com/Haoming02/sd-forge-couple) | Mutates `p.prompt` and `p.all_prompts` during `process()`, which breaks line-based or bracket-based couple separators in `sd-forge-couple`[44](https://github.com/Haoming02/sd-forge-couple). | **Structured CRF Immunity**: Because the extension stores regional prompts inside a structured JSON state (`p.extra_generation_params["crf_spec"]`) rather than relying on fragile newline counting in the main prompt box[44](https://github.com/Haoming02/sd-forge-couple), wildcard expansion is explicitly invoked on each region's prompt string *before* sequential text encoding. |
| **`NegPiP` (Negative Prompt in Prompt)**[68](https://wikiwiki.jp/sd_toshiaki/Forge%20neo) | Modifies `K` and `V` cross-attention tensors to invert weights; throws tensor shape errors when combined with `Forge Couple` in `Attention` mode[68](https://wikiwiki.jp/sd_toshiaki/Forge%20neo). | **Hook Chaining**: Apply `NegPiP` token weight inversion during the regional text-encoding stage (when generating \(\mathbf{K}_r, \mathbf{V}_r\)) rather than inside the runtime `attn2` replacement hook, preserving exact tensor dimensions. |
| **`ADetailer` (Post-Processing)**[67](https://github.com/lllyasviel/stable-diffusion-webui-forge/issues) | Runs `img2img` inpainting on detected faces/hands during `postprocess()`. In Forge Neo, if high-res hooks or custom attributes remain attached to `sd_model`, `ADetailer` throws an `AttributeError`[67](https://github.com/lllyasviel/stable-diffusion-webui-forge/issues). | **Strict `postprocess()` Teardown**: Execute complete `ModelPatcher` unpatching and restore original `Original_Size` / `RoPE` / `RAU-Net` state inside `postprocess_batch()` *before* `ADetailer`'s `postprocess_image()` hook fires, ensuring `ADetailer` sees a clean, stock \(1024 \times 1024\) model[6](https://github.com/GumGum10/forge-anima-3.8B). |
| **`ControlNet` & `LLLite`**[2](https://github.com/Haoming02/sd-webui-forge-classic) | Uses `set_model_unet_function_wrapper` and `input_block_patch`. | **Cooperative Wrapper Chaining**: Never overwrite an existing `model_function_wrapper`; wrap the existing callable (`prev_wrapper(x, timestep, **c)`) and slice `c["control"]` / `LLLite` cond tensors using the exact current tile coordinates \([y_0:y_1, x_0:x_1]\)[2](https://github.com/Haoming02/sd-webui-forge-classic). |
| **Exception & Interruption Handling**[44](https://github.com/Haoming02/sd-forge-couple) | In Forge's `call_queue.py`, raising an unhandled exception inside an extension script is caught and logged while the main sampler continues running with unpatched/corrupted tensors, causing "silent failures"[44](https://github.com/Haoming02/sd-forge-couple). | **Explicit `state.interrupt()` + `try...finally` Restoration**: Wrap all `process()` and `unet_wrapper` hooks in a `try...except Exception as e:` block that logs a formatted traceback, calls `modules.shared.state.interrupt()` to halt the sampler cleanly (matching `sd-forge-couple`'s safeguard)[44](https://github.com/Haoming02/sd-forge-couple), and executes a `finally:` block that restores `ModelPatcher` weights and frees pinned CPU buffers[6](https://github.com/GumGum10/forge-anima-3.8B). |
| **Batch Generation, Seeds & Infotext**[6](https://github.com/GumGum10/forge-anima-3.8B) | Multi-pass or `Latent` mode LoRA switching can corrupt RNG state across batch items, causing identical seeds to yield different images on repeat clicks[47](https://github.com/hako-mikan/sd-webui-regional-prompter/issues/400). | **Deterministic Per-Stage Seed Offsetting & Infotext Serialization**: Re-seed `torch.Generator` deterministically at the start of each progressive stage (`seed + stage_idx * 10007`). Serialize the compressed CRF JSON (or SHA-256 reference + inline regions) into PNG `infotext` (`CRF Specification: {...}`) so dragging a generated PNG into Forge Neo restores the entire multilayer canvas, regional prompts, and LoRAs losslessly[6](https://github.com/GumGum10/forge-anima-3.8B). |

#### Legal Code Reuse and Open-Source Licensing Audit

Before integrating external algorithms, every reference repository's license was audited to prevent copyright infringement or license contamination:

| Reference Project / Algorithm | Repository / Source | Verified License | Legal Code Reuse Status & Integration Conditions |
| --- | --- | --- | --- |
| **Forge Neo Core & `sd-forge-couple`** | `Haoming02/sd-webui-forge-classic` (`neo`) & `Haoming02/sd-forge-couple`[2](https://github.com/Haoming02/sd-webui-forge-classic) | **GPL-3.0** (Inherited from A1111 / Forge) | **Direct Reuse Permitted** (if extension is distributed under GPL-3.0 or MIT/Apache-2.0 compatible terms). Reuse hook registration patterns and subject-replacement parsing[44](https://github.com/Haoming02/sd-forge-couple). |
| **DemoFusion** | `PRIS-CV/DemoFusion`[12](https://github.com/PRIS-CV/DemoFusion) | **MIT License**[12](https://github.com/PRIS-CV/DemoFusion) | **Unrestricted Direct Reuse**. Can directly adapt progressive skip-residual cosine scaling (`cosine_scale_1/2/3`), dilated sampling stride logic, and Gaussian filtering[12](https://github.com/PRIS-CV/DemoFusion). Retain copyright notice. |
| **DemoFusion WebUI Port** | `sebaxakerhtc/sd-webui-demofusion`[71](https://github.com/sebaxakerhtc/sd-webui-demofusion) | **Apache-2.0 License**[71](https://github.com/sebaxakerhtc/sd-webui-demofusion) | **Direct Reuse Permitted**. Provides reference for A1111/Forge VAE multidecoder and custom resolution UI wiring[71](https://github.com/sebaxakerhtc/sd-webui-demofusion). |
| **HiDiffusion & Forge Ports** | `megvii-research/HiDiffusion`, `MisterChief95/sd-forge-hidiffusion` (from `Panchovix` / `blepping`)[41](https://github.com/megvii-research/HiDiffusion) | **Apache-2.0 License** (Vendored in InvokeAI under `LICENSE-HiDiffusion.txt`)[41](https://github.com/megvii-research/HiDiffusion) | **Direct Reuse Permitted**. Directly reuse the `ModelPatcher`-compatible `RAU-Net` (`RAD`/`RAU`) and `MSW-MSA` implementations from `sd-forge-hidiffusion`, retaining the Apache-2.0 `NOTICE` header[42](https://github.com/MisterChief95/sd-forge-hidiffusion). |
| **UltraDiffEdit (2026)** | `LonglongaaaGo/UltraDiffEdit`[76](https://github.com/LonglongaaaGo/UltraDiffEdit) | **Apache-2.0 License**[76](https://github.com/LonglongaaaGo/UltraDiffEdit) | **Direct Reuse Permitted**. Coarse-to-fine latent blending (\(\hat{\mathbf{z}}_t^s\)) and boundary consistency logic can be integrated directly[40](https://arxiv.org/html/2607.06136v1). |
| **ScaleCrafter** | `yingqinghe/scalecrafter`[77](https://syntology.ai/paper/2310.07702) | **NONE (All Rights Reserved / Unlicensed)**[77](https://syntology.ai/paper/2310.07702) | **DO NOT COPY CODE**. GitHub detects no license on `yingqinghe/scalecrafter` (code withheld by compliance harvesters)[77](https://syntology.ai/paper/2310.07702). Since ScaleCrafter is also inferior to HiDiffusion, exclude it entirely[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf). |
| **AccDiffusion v1 / v2** | `lzhxmu/AccDiffusion` & `lzhxmu/AccDiffusion_v2`[8](https://arxiv.org/pdf/2412.02099?) | **Clean-Room Implementation** | Implement the mathematical equations (Eq. 9 binary cross-attention mask \(B_{i,j}\) and position-wise bijection permutation) cleanly from the published IEEE/ECCV papers[3](https://lzhxmu.github.io/accdiffusion/accdiffusion.html). |
| **Anima Base & 3.8B Extension** | `circlestone-labs/Anima` & `GumGum10/forge-anima-3.8B`[6](https://github.com/GumGum10/forge-anima-3.8B) | **CircleStone Labs Model License** (built on NVIDIA Cosmos)[49](https://civitai.com/models/2604674/controlnet-lllite-anima-base-v1) | Model weights are governed by CircleStone Labs LLC terms[49](https://civitai.com/models/2604674/controlnet-lllite-anima-base-v1). The extension does not bundle model weights; it interfaces with `forge-anima-3.8B` and native Forge Neo Anima loaders via runtime hooks[6](https://github.com/GumGum10/forge-anima-3.8B). |

## Deliverable 3: Consolidated Functional and Multilayer Canvas UX Requirements (PRODUCT-REQUIREMENTS.md)

### Track 6: Regional Canvas UX Research and Frontend Architecture

Existing regional conditioning interfaces in the A1111/Forge ecosystem suffer from severe usability defects. `Regional Prompter` forces users to write fragile `ADDROW` / `ADDCOL` / `BREAK` syntax strings or upload external color-coded masks ("Upload mask here cus gradio", clicking twice on "Draw Region")[18](https://github.com/hako-mikan/sd-webui-regional-prompter). `sd-forge-couple` improves on this with an `Advanced Mode` bounding-box table and a `Mask Mode`, but its `Mask Mode` relies on Gradio's default single-layer sketchpad: users must draw one region, save it to a list, clear the canvas, and draw the next region blindly without seeing existing regions simultaneously, or manually type normalized floats `[0.0, 1.0]` into a table where accidental desynchronization with the main prompt field causes silent errors or generation interrupts[44](https://github.com/Haoming02/sd-forge-couple).

To replace this draw–save–load bottleneck with a production-grade, non-destructive multilayer visual editor inside Forge Neo's Gradio 4.40.0 frontend[2](https://github.com/Haoming02/sd-webui-forge-classic), mature web canvas engines and creative tools (InvokeAI Unified Canvas[74](https://github.com/invoke-ai/InvokeAI?ref=blindegg.com), Figma, Krita/Photopea, Konva.js, Fabric.js, Paper.js, and PixiJS[79](https://konvajs.org/docs/faq.html)) were systematically evaluated.

#### Evaluation of Web Canvas Engines for Gradio 4.40.0 Integration

| Canvas Engine | Rendering Architecture | Multilayer & Hit Detection | Vector vs. Freehand Raster Tools | Serialization & State Sync | Suitability for Forge Neo Regional Canvas |
| --- | --- | --- | --- | --- | --- |
| **Konva.js**[79](https://konvajs.org/docs/faq.html) | **Multi-Canvas HTML5 2D** (Each `Konva.Layer` is a separate `<canvas>` DOM element)[79](https://konvajs.org/docs/faq.html). | **Best-in-Class**: Moving or painting an active layer never triggers re-renders on static background/sibling layers[80](https://konvajs.org/docs/guides/best-canvas-library.html). Built-in `Transformer` resize/rotate handles, event bubbling, and pixel-accurate hit detection[80](https://konvajs.org/docs/guides/best-canvas-library.html). | Native support for Rectangles (`Konva.Rect`), Polygons (`Konva.Line` with closed paths and vertex anchors), and freehand brush/eraser via `globalCompositeOperation` on dedicated offscreen layer canvases[80](https://konvajs.org/docs/guides/best-canvas-library.html). | Built-in `stage.toJSON()` and `Konva.Node.create(json)` for instant state persistence across Gradio reloads[80](https://konvajs.org/docs/guides/best-canvas-library.html). | **Selected Primary Engine**. Multi-canvas architecture guarantees 60 FPS responsiveness at 4K/8K aspect ratios while keeping all 5+ regions visible simultaneously[79](https://konvajs.org/docs/faq.html). |
| **Fabric.js**[79](https://konvajs.org/docs/faq.html) | **Single-Canvas HTML5 2D** (All objects re-render on a single `<canvas>`)[79](https://konvajs.org/docs/faq.html). | Basic object stacking (`z-index`), but modifying one complex brush stroke forces full-canvas redraw[79](https://konvajs.org/docs/faq.html). | Strong built-in drawing brushes and SVG export, but lacks Konva's isolated multi-canvas layer performance[79](https://konvajs.org/docs/faq.html). | JSON/SVG serialization, requires custom state reconstruction for complex multi-mask compositing[82](https://img.ly/fabricjs-alternative/). | **Secondary Alternative**. Single-canvas redraw degrades when compositing overlapping semi-transparent raster masks + heatmaps[79](https://konvajs.org/docs/faq.html). |
| **Paper.js**[79](https://konvajs.org/docs/faq.html) | Scene-graph Vector HTML5 2D. | Good vector grouping, weaker raster brush performance. | **Best for Boolean Path Math** (union, subtract, intersect of polygons and Bezier curves)[79](https://konvajs.org/docs/faq.html). | Custom JSON export. | **Utility Library Only** (can use lightweight polygon clipping math if needed, not for main UI)[80](https://konvajs.org/docs/guides/best-canvas-library.html). |
| **PixiJS**[79](https://konvajs.org/docs/faq.html) | WebGL GPU-Accelerated 2D Engine[79](https://konvajs.org/docs/faq.html). | High frame rate for thousands of moving sprites[79](https://konvajs.org/docs/faq.html). | Lacks built-in UI manipulation handles (`Transformer`), requiring building resize handles and polygon vertex editors from scratch[80](https://konvajs.org/docs/guides/best-canvas-library.html). | Requires custom serialization and consumes client browser GPU contexts (competing with WebUI VRAM). | **Rejected**: Consumes GPU VRAM for WebGL contexts and lacks out-of-the-box design editor handles[79](https://konvajs.org/docs/faq.html). |

#### Consolidated Functional & UX Requirements (`PRODUCT-REQUIREMENTS.md`)

To satisfy the multilayer visualization requirements, the editor mounts a custom JavaScript/TypeScript `Konva.Stage` inside Forge Neo's extension accordion[80](https://konvajs.org/docs/guides/best-canvas-library.html). The stage is architected around four stacked layer groups that isolate static background elements from active manipulation handles[79](https://konvajs.org/docs/faq.html). At the bottom sits the *Base Reference Layer*, which renders an optional background image (such as a Stage 1 anchor preview, an uploaded composition reference, or a ControlNet map) alongside a toggleable tile-grid overlay indicating exact \(1024 \times 1024\) tile boundaries and overlap zones. Directly above this sit the *Region Vector & Raster Layers (\(1 \dots R\))*, where each region is instantiated as an independent `Konva.Layer` assigned a distinct, high-contrast OKLCH/HSL color from a colorblind-safe categorical palette. Both global and per-layer opacity sliders (\(\alpha_{\text{view}} \in [0.10, 0.90]\), defaulting to `0.45`) ensure that all regions remain simultaneously visible while any individual region is drawn or edited[80](https://konvajs.org/docs/guides/best-canvas-library.html). The third group is the *Effective Conditioning Heatmap Overlay Layer*, a toggleable real-time visualization layer that computes and renders the exact normalized conditioning weight \(\overline{M}_r(x, y)\) at every pixel[22](https://github.com/pamparamm/ComfyUI-ppm/issues/21), highlighting overlap conflict zones with a split-color gradient and displaying a live hover tooltip (for example, `Pixel (2048, 1536): Global 20% | Sky 40% | Sun 40%`)[18](https://github.com/hako-mikan/sd-webui-regional-prompter). At the top of the stack sits the *Active Interaction & Transformer Layer*, which hosts the `Konva.Transformer` bounding-box handles, polygon vertex drag nodes, smart snapping guides, and freehand brush cursors[80](https://konvajs.org/docs/guides/best-canvas-library.html).

Non-destructive multi-tool creation and editing are achieved by unifying vector primitives and raster mask buffers within the same layer tree. Vector tools—specifically the Bounding Box tool (`B`) and Polygon/Lasso tool (`P`)—store region geometry mathematically as normalized coordinate arrays (`bbox: [x, y, w, h]` or `polygon: [[x1, y1], [x2, y2], ...]`), meaning that selecting an existing region activates its resize handles or vertex nodes *in-place* without rasterizing, erasing, or concealing any sibling layer[80](https://konvajs.org/docs/guides/best-canvas-library.html). For organic masks, the Raster Freehand tools—Brush (`D`), Eraser (`E`), Magic Wand / Flood Fill (`W`), and a live Feather/Blur radius slider—operate on a dedicated offscreen mask canvas bound strictly to the currently selected `active_layer_id`, leaving all other layers untouched. Precision viewport navigation includes mouse-wheel zoom (`10%` to `800%`), middle-click or `Space`+drag panning, magnetic snapping to canvas edges, centerlines, and sibling region borders (toggleable via `Shift`), a 50-state undo/redo history stack persisted in browser `IndexedDB` / `localStorage` to survive accidental Gradio UI reloads[80](https://konvajs.org/docs/guides/best-canvas-library.html), and direct numeric coordinate entry fields (`X, Y, W, H` in either normalized `0.000–1.000` floats or target pixels `0–4096`).

Bi-directional binding connects the visual canvas directly to conditioning parameters. Clicking a region shape on the canvas highlights its corresponding Layer Card in the right-hand Layer Inspector Panel, and selecting a Layer Card highlights and focuses the region on the canvas[44](https://github.com/Haoming02/sd-forge-couple). Each Layer Card encapsulates layer management toggles (`Visible` eye icon, `Lock` padlock icon, `Z-Order` drag handle, `Rename`, `Duplicate`, `Delete`, and a `Weight` slider spanning \(0.0\text{–}5.0\) with default \(1.0\)) alongside dedicated `Positive Prompt` and `Negative Prompt` textareas bound specifically to that region's `id`, eliminating `sd-forge-couple`'s error-prone newline synchronization with the main prompt box[44](https://github.com/Haoming02/sd-forge-couple). Each card also integrates a **Regional LoRA Chip Selector**, enabling users to attach one or more LoRAs (`name`, `weight`, `te_weight`, and `isolation_tier`) directly to the region without embedding `<lora:...>` tags inside prompt strings[2](https://github.com/Haoming02/sd-webui-forge-classic).

#### Usability Acceptance Test Protocol

To verify that the editor eliminates legacy draw–save–load limitations, the frontend must pass the **5-Region Non-Destructive Edit Acceptance Test**. In this test, an operator creates five overlapping regions (`Sky` bounding box, `Sun` polygon, `Ocean` bounding box, `Lighthouse` freehand brush mask, and `Beach` bounding box) on a \(4096 \times 3072\) canvas and subsequently selects Region 2 (`Sun`) via the layer list or canvas click. To pass, Regions 1, 3, 4, and 5 must remain continuously visible on the canvas with their distinct colors and labels while the operator drags a polygon vertex of `Sun` and adjusts its regional LoRA weight from `0.6` to `0.8`. Simultaneously, the Effective Conditioning Heatmap must update at 60 FPS in the overlap zone between `Sky` and `Sun`, zero pixels of Regions 1, 3, 4, or 5 may be modified, hidden, or re-uploaded, and triggering Undo (`Ctrl+Z`) must revert the vertex translation in `<16 ms`.

## Deliverable 4: Coherent Regions Format Specification and Schema (CRF-SPEC.md)

### Track 7: CRF Communication Protocol, JSON Schema, and LLM Integration

To enable external Large Language Models (LLMs), automated storyboarding pipelines, and third-party UIs (such as ComfyUI or InvokeAI) to exchange complex regional compositions losslessly with Forge Neo, the provisional `coherent-regions` draft is formalized into the **Coherent Regions Format (CRF) v1 Specification**.

#### Engineering Resolution of Open Protocol Questions

| Protocol Dimension | Formal Specification & Deterministic Semantics |
| --- | --- |
| **1. Coordinate Space & Aspect-Ratio Divergence** | Origin \((0.0, 0.0)\) is strictly the **top-left corner** of the canvas; \((1.0, 1.0)\) is the bottom-right. All spatial coordinates (`bbox: [x, y, width, height]` and `polygon: [[x, y], ...]`) are normalized floats in \([0.0, 1.0]\). When importing a CRF document whose `canvas.width / canvas.height` aspect ratio differs from Forge Neo's current target dimensions by \(>1\%\), the importer displays a non-blocking **Aspect-Ratio Mismatch Warning Banner** and offers three deterministic mapping modes: `stretch` (default: preserves normalized \([0,1]\) coordinates directly), `letterbox_fit` (scales regions uniformly and pads the background with `global`), or `adopt_crf_canvas` (updates Forge Neo's width/height sliders to match `canvas`). |
| **2. Region Geometry Types** | Every region specifies `"shape_type": "bbox" |
| **3. Overlapping Mask Semantics & Blending** | Regions are evaluated in ascending order of their integer `"z_index"` (or array order if `"z_index"` is omitted). Each region specifies an `"overlap_mode"` (`"weighted_blend"` [default] or `"occlude"`): • `"weighted_blend"`: In intersecting areas, active region masks are multiplied by their region `"weight"` \(w_r \in [0.0, 5.0]\) and \(L_1\)-normalized per pixel: \(\tilde{M}_r(i) = \frac{w_r M_r(i)}{\sum_{k} w_k M_k(i)}\) (matching `AttentionCouplePPM` and `sd-forge-couple`)[22](https://github.com/pamparamm/ComfyUI-ppm/issues/21). • `"occlude"`: Higher `z_index` regions subtract their alpha mask from underlying regions (\(M_{\text{lower}} \leftarrow M_{\text{lower}} \cdot (1 - M_{\text{upper}})\)), enabling crisp foreground cutouts (e.g., a character standing in front of a detailed background without background prompt bleed). |
| **4. Global Prompt Inheritance & Uncovered Pixels** | To prevent `sd-forge-couple`'s fatal error (*"Image must contain weights on all pixels"*)[44](https://github.com/Haoming02/sd-forge-couple), any pixel where the sum of regional masks \(\sum_r M_r(i) = 0\) automatically receives \(100\%\) of the `"global"` prompt conditioning. Inside covered regions, `"global"` prompt inheritance is governed by `"global_inheritance"` (`"base_ratio"` float in \([0.0, 1.0]\), defaulting to `0.20` per `Regional Prompter` best practices[18](https://github.com/hako-mikan/sd-webui-regional-prompter), plus optional `"prepend_global_text": false`). Thus, effective regional attention is \(\text{ base\_ratio} \cdot \text{Global} + (1 - \text{base\_ratio}) \cdot \text{Regional}\)[18](https://github.com/hako-mikan/sd-webui-regional-prompter). |
| **5. Portable LoRA References & Isolation Tiers** | Each LoRA entry in `"loras"` specifies `"name"` (portable stem filename without path/extension), `"weight"` (U-Net/DiT strength, default `1.0`), optional `"te_weight"` (text-encoder strength), optional `"sha256"` (first 10 or full 64 hex chars of Civitai/safetensors AutoV2/V3 hash for exact portability across renamed files[6](https://github.com/GumGum10/forge-anima-3.8B)), and optional `"isolation_tier": "masked_delta" |
| **6. Security Constraints** | The CRF parser uses strict `json.loads()` validated against Draft 2020-12 JSON Schema (never `eval()`). All `"mask_uri"` paths inside `.crf.zip` bundles are sanitized against path traversal (`..`, absolute paths `/`, `\`, or external `http://` URLs are rejected). LoRA `"name"` strings are matched strictly against Forge Neo's indexed `extra_networks` registry; arbitrary filesystem paths are rejected. |

#### Machine-Readable JSON Schema (`crf-v1.schema.json`)

The complete Draft 2020-12 JSON Schema definition below enforces strict structural validation, coordinate bounds checking, and security sanitization for all CRF v1 payloads.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://forgeneo.org/schemas/coherent-regions-v1.schema.json",
  "title": "Coherent Regions Format (CRF)",
  "description": "Resolution-independent spatial conditioning and regional LoRA interchange format for Forge Neo and external LLM workflows.",
  "type": "object",
  "required": ["format", "version", "canvas", "global", "regions"],
  "additionalProperties": false,
  "properties": {
    "format": {
      "type": "string",
      "const": "coherent-regions"
    },
    "version": {
      "type": "integer",
      "enum": [1]
    },
    "canvas": {
      "type": "object",
      "required": ["width", "height"],
      "additionalProperties": false,
      "properties": {
        "width": { "type": "integer", "minimum": 256, "maximum": 16384 },
        "height": { "type": "integer", "minimum": 256, "maximum": 16384 },
        "aspect_mismatch_policy": {
          "type": "string",
          "enum": ["stretch", "letterbox_fit", "adopt_crf_canvas"],
          "default": "stretch"
        }
      }
    },
    "global": {
      "type": "object",
      "required": ["prompt"],
      "additionalProperties": false,
      "properties": {
        "prompt": { "type": "string" },
        "negative_prompt": { "type": "string", "default": "" },
        "base_ratio": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0,
          "default": 0.2,
          "description": "Fraction of global conditioning blended into every regional mask."
        },
        "loras": {
          "type": "array",
          "items": { "$ref": "#/$defs/lora_binding" },
          "default": []
        }
      }
    },
    "synthesis_options": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "mode": {
          "type": "string",
          "enum": ["progressive_hybrid", "direct_hidiffusion", "tiled_multidiffusion"],
          "default": "progressive_hybrid"
        },
        "auto_patch_prompts": { "type": "boolean", "default": true },
        "anima_region_controlnet": { "type": "boolean", "default": false }
      }
    },
    "regions": {
      "type": "array",
      "minItems": 1,
      "maxItems": 64,
      "items": { "$ref": "#/$defs/region_item" }
    }
  },
  "$defs": {
    "lora_binding": {
      "type": "object",
      "required": ["name", "weight"],
      "additionalProperties": false,
      "properties": {
        "name": {
          "type": "string",
          "pattern": "^[a-zA-Z0-9_\\-\\. ]+$",
          "description": "Portable LoRA identifier without directory paths."
        },
        "weight": { "type": "number", "minimum": -4.0, "maximum": 4.0 },
        "te_weight": { "type": "number", "minimum": -4.0, "maximum": 4.0 },
        "sha256": { "type": "string", "pattern": "^[a-fA-F0-9]{10,64}$" },
        "isolation_tier": {
          "type": "string",
          "enum": ["masked_delta", "strict_attn_mask", "latent_pass"],
          "default": "masked_delta"
        }
      }
    },
    "region_item": {
      "type": "object",
      "required": ["id", "name", "prompt"],
      "additionalProperties": false,
      "properties": {
        "id": { "type": "string", "pattern": "^[a-zA-Z0-9_\\-]+$" },
        "name": { "type": "string", "maxLength": 128 },
        "enabled": { "type": "boolean", "default": true },
        "locked": { "type": "boolean", "default": false },
        "z_index": { "type": "integer", "minimum": 0, "maximum": 1000, "default": 0 },
        "shape_type": {
          "type": "string",
          "enum": ["bbox", "polygon", "svg_path", "raster_ref"],
          "default": "bbox"
        },
        "bbox": {
          "type": "array",
          "items": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
          "minItems": 4,
          "maxItems": 4,
          "description": "Normalized [x, y, width, height] with top-left origin."
        },
        "polygon": {
          "type": "array",
          "minItems": 3,
          "items": {
            "type": "array",
            "items": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
            "minItems": 2,
            "maxItems": 2
          }
        },
        "svg_path": { "type": "string" },
        "mask_uri": {
          "type": "string",
          "pattern": "^masks/[a-zA-Z0-9_\\-]+\\.png$"
        },
        "feather_radius": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 0.2,
          "default": 0.02,
          "description": "Normalized Gaussian blur radius at mask edges."
        },
        "weight": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 5.0,
          "default": 1.0
        },
        "overlap_mode": {
          "type": "string",
          "enum": ["weighted_blend", "occlude"],
          "default": "weighted_blend"
        },
        "base_ratio_override": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0
        },
        "prompt": { "type": "string" },
        "negative_prompt": { "type": "string", "default": "" },
        "loras": {
          "type": "array",
          "items": { "$ref": "#/$defs/lora_binding" },
          "default": []
        }
      },
      "anyOf": [
        { "required": ["bbox"] },
        { "required": ["polygon"] },
        { "required": ["svg_path"] },
        { "required": ["mask_uri"] }
      ]
    }
  }
}
```

#### Complete Validated CRF v1 Composition Example (Demonstrating Overlap, Polygons, and Occlusion)

The following documented example illustrates how CRF v1 extends the initial coastal landscape draft to incorporate explicit `z_index` ordering, foreground `occlude` semantics for the sun and cliffside lighthouse, polygon vertices, and regional LoRA isolation tiers.

```json
{
  "format": "coherent-regions",
  "version": 1,
  "canvas": {
    "width": 4096,
    "height": 3072,
    "aspect_mismatch_policy": "stretch"
  },
  "global": {
    "prompt": "Professional landscape photograph, natural colors, realistic lighting, extraordinary photographic detail",
    "negative_prompt": "CGI, illustration, distortion, watermark",
    "base_ratio": 0.20
  },
  "synthesis_options": {
    "mode": "progressive_hybrid",
    "auto_patch_prompts": true,
    "anima_region_controlnet": false
  },
  "regions": [
    {
      "id": "sky",
      "name": "Sky",
      "z_index": 10,
      "shape_type": "bbox",
      "bbox": [0.0, 0.0, 1.0, 0.45],
      "weight": 1.0,
      "overlap_mode": "weighted_blend",
      "prompt": "Clear blue atmospheric sky, wispy cirrus clouds"
    },
    {
      "id": "sun",
      "name": "Sun",
      "z_index": 20,
      "shape_type": "bbox",
      "bbox": [0.70, 0.08, 0.12, 0.16],
      "feather_radius": 0.03,
      "weight": 1.5,
      "overlap_mode": "occlude",
      "base_ratio_override": 0.10,
      "prompt": "One small bright sun, localized atmospheric glow",
      "negative_prompt": "multiple suns, lens artifacts"
    },
    {
      "id": "ocean",
      "name": "Ocean",
      "z_index": 15,
      "shape_type": "bbox",
      "bbox": [0.0, 0.45, 1.0, 0.35],
      "weight": 1.0,
      "overlap_mode": "weighted_blend",
      "prompt": "Ocean horizon, distant waves, golden sunlight reflections on water"
    },
    {
      "id": "lighthouse_cliff",
      "name": "Rocky Cliff & Lighthouse",
      "z_index": 30,
      "shape_type": "polygon",
      "polygon": [
        [0.0, 0.32],
        [0.24, 0.32],
        [0.28, 0.78],
        [0.0, 0.78]
      ],
      "feather_radius": 0.015,
      "weight": 1.4,
      "overlap_mode": "occlude",
      "prompt": "Single weathered stone lighthouse standing on a rugged coastal cliff, crashing sea spray",
      "negative_prompt": "multiple lighthouses, modern buildings"
    },
    {
      "id": "beach",
      "name": "Beach",
      "z_index": 25,
      "shape_type": "bbox",
      "bbox": [0.0, 0.80, 1.0, 0.20],
      "weight": 1.0,
      "overlap_mode": "weighted_blend",
      "prompt": "Natural sandy beach, wet shoreline, realistic coastal textures",
      "loras": [
        {
          "name": "example_texture_lora",
          "weight": 0.6,
          "isolation_tier": "masked_delta"
        }
      ]
    }
  ]
}
```

#### Compact Instruction Template for External LLMs

To enable any external LLM to emit valid CRF v1 JSON reliably on the first attempt, applications can provide the following concise system instruction block:

SYSTEM INSTRUCTION — COHERENT REGIONS FORMAT (CRF v1) GENERATOR Output ONLY a single valid JSON object conforming to the Coherent Regions Format (CRF v1). Rules:

1. Top-level keys required: "format": "coherent-regions", "version": 1, "canvas": {"width": int, "height": int}, "global": {"prompt": str, "negative_prompt": str, "base_ratio": 0.2}, and "regions": [...].
2. Coordinate system: Normalized floats in [0.0, 1.0] with (0.0, 0.0) at the top-left corner.
3. For rectangular regions, set "shape_type": "bbox" and "bbox": [x, y, width, height] where x + width <= 1.0 and y + height <= 1.0.
4. For non-rectangular regions, set "shape_type": "polygon" and "polygon": [[x1, y1], [x2, y2], [x3, y3], ...].
5. Order background layers with lower "z_index" (e.g., 10) and foreground subjects placed inside background areas with higher "z_index" (e.g., 20, 30) and "overlap_mode": "occlude".
6. Keep regional "prompt" strings focused strictly on the single subject or material inside that region. Put overall lighting, camera lens, and art style in "global.prompt". Add negative counters (e.g., "multiple suns") to "negative_prompt" for unique objects.

#### CRF Conformance and Validation Test Suite

| Test ID | Test Input Scenario | Expected Parser & Validator Outcome |
| --- | --- | --- |
| **CRF-T01** | Valid `bbox` and `polygon` CRF v1 JSON (as in the 5-region example above). | Passes JSON Schema validation; renders preview overlay; populates 5 Konva layers with exact `z_index`, prompts, and LoRAs. |
| **CRF-T02** | Out-of-bounds bbox `bbox: [0.75, 0.10, 0.40, 0.20]` (\(x + w = 1.15 > 1.0\)). | Validator catches spatial overflow; emits structured warning `Region 'sun' x+width (1.15) exceeds 1.0; clamped to [0.75, 0.10, 0.25, 0.20]`. |
| **CRF-T03** | Uncovered canvas gap (regions cover only \(y \in [0.0, 0.70]\), leaving bottom \(30\%\) empty). | Validator auto-fills uncovered pixels \([0.0, 0.70, 1.0, 0.30]\) with \(100\%\) `global.prompt` conditioning, preventing `sd-forge-couple` zero-weight crashes[44](https://github.com/Haoming02/sd-forge-couple). |
| **CRF-T04** | Malicious path traversal `"mask_uri": "../../Windows/win.ini"` or `"name": "../evil_lora"`. | Rejected by regex pattern validation before touching the filesystem; logs security error and aborts import safely. |
| **CRF-T05** | Round-trip serialization (`Import CRF` \(\rightarrow\) operator moves Region 2 by `+0.05` on canvas \(\rightarrow\) `Export CRF`). | Exported JSON preserves all comments/metadata, `id`s, `z_index`, prompts, and LoRAs losslessly, updating only Region 2's coordinates. |

## Deliverable 5: Objective Evaluation and Reproducible Benchmarking Protocol (BENCHMARKS.md)

### Track 10: Reproducible Test Suite, Automated Metrics, and Human Evaluation

To prevent subjective architectural decisions during development, all algorithmic candidates and memory optimizations must be evaluated against a deterministic, multi-seed benchmark suite prior to merging into the main branch.

#### Standardized Four-Category Compositional Benchmark Suite

Every candidate pipeline must be evaluated across 5 fixed seeds (`[42, 1337, 2026, 8888, 99999]`) at three target resolutions (\(1024 \times 1024\) native reference, \(2048 \times 2048\), and \(4096 \times 3072\) / \(4096 \times 4096\)) on both **SDXL 1.0 Base** and **Anima 3.8B v2**[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf):

| Benchmark Case ID | Compositional Stress Target | Standardized CRF / Global Prompt Test Case | Primary Failure Modes Tested |
| --- | --- | --- | --- |
| **BM-01: Unique Objects** | Preventing small and large object duplication across tiles[3](https://lzhxmu.github.io/accdiffusion/accdiffusion.html). | **4K Coastal Scene** (5-region CRF: Sky, 1 small bright sun, Ocean, 1 stone lighthouse on cliff, Sandy beach with texture LoRA). Tested **both** with CRF masks and as a raw global prompt (testing Track 4 auto-decomposition)[8](https://arxiv.org/pdf/2412.02099?). | Multiple suns in sky; phantom lighthouses in water; broken horizon step across vertical tile seams[3](https://lzhxmu.github.io/accdiffusion/accdiffusion.html). |
| **BM-02: Subject Consistency & LoRA Isolation** | Zero attribute/identity bleed between two conflicting character/style LoRAs[20](https://arxiv.org/pdf/2407.02329?). | **Dual-Character Tavern Scene** (\(3072 \times 2048\)): Left region (`[0.05, 0.15, 0.40, 0.80]`) = Character A (`cyan twintails, white serafuku` + `CharA_LoRA:0.9`); Right region (`[0.55, 0.15, 0.40, 0.80]`) = Character B (`black long hair, red eyes, dark armor` + `CharB_LoRA:0.9`)[44](https://github.com/Haoming02/sd-forge-couple). | Hair/eye color swapping; LoRA style contaminating the opposing character or tavern background; extra limbs/merged anatomy[21](https://github.com/hako-mikan/sd-webui-regional-prompter/discussions/272). |
| **BM-03: Architectural Geometry & Diagonals** | Preserving vanishing points, straight perspective lines, and diagonal structures across 2D RoPE / U-Net tiles[16](https://arxiv.org/html/2508.15774v2). | **Grand Gothic Cathedral Interior** (\(2048 \times 3072\) portrait): Vaulted diagonal rib arches converging at a central stained-glass rose window, checkered marble floor tiles receding in perspective. | Axial 2D RoPE diagonal distortion[39](https://medium.com/@cenghanbayram35/spiral-rope-vision-transformers-finally-learn-to-see-diagonals-f38683609173); cloned rose windows in upper corners (U-Net deep feature duplication)[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf); broken floor tile perspective lines. |
| **BM-04: Global Continuity & Micro-Texture** | Seamless lighting, terrain gradients, and absence of VAE grid artifacts[11](https://arxiv.org/html/2306.05178v3). | **Alpine Valley Panorama** (\(4096 \times 2048\)): Continuous winding river reflecting sunset gradient from warm orange on left to deep indigo on right, detailed pine needles and foreground snow textures. | MultiDiffusion lighting jumps between tiles[11](https://arxiv.org/html/2306.05178v3); `qwen_image_vae` checkerboard/grid artifacts[27](https://github.com/MoayedHajiAli/ElasticDiffusion-official); over-smoothed textures from excessive skip-residual weight[12](https://github.com/PRIS-CV/DemoFusion). |

#### Quantitative and Human Evaluation Metrics

Standard full-image FID and CLIP Score alone fail to penalize localized object duplication; as demonstrated in the FreeScale study, SDXL direct inference achieves deceptively strong cropped FID/KID scores (\(\text{FID}_c / \text{KID}_c\)) precisely *because* it generates multiple duplicated objects whose small crops resemble the training distribution, while suffering the worst full-image FID and KID at \(4096^2\)[9](https://arxiv.org/html/2412.09626v1). Consequently, the benchmark harness enforces a six-dimensional evaluation pipeline combining automated spatial metrics, hardware telemetry, and blind human evaluation.

Distribution and detail fidelity are assessed by pairing full-image `FID` and `KID` (which measure global structural realism) with **Patches FID (`pFID`)**[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf) and spatial FID (`sFID`)[84](https://arxiv.org/html/2503.18719v1), which evaluate high-frequency local detail on \(1024 \times 1024\) crops without downsampling. To directly quantify object repetition, **Automated Object-Count Accuracy (`OCA`)** executes an open-vocabulary detector (`GroundingDINO` / `OWL-ViT`) in a separate post-generation evaluation process on the finished 4K output to count detected instances of target nouns (`"sun"`, `"lighthouse"`, `"person"`), computing \(\text{OCA} = 1 - \frac{|N_{\text{detected}} - N_{\text{expected}}|}{\max(N_{\text{detected}}, N_{\text{expected}})}\). Cross-region contamination is measured via the **Regional LoRA Isolation Score (`RLIS`)**, which computes the cosine similarity of CLIP and DINOv2 embeddings of Region A's crop against an isolated single-subject reference generation of `LoRA_A` versus `LoRA_B`, thereby quantifying attribute leakage across masks[20](https://arxiv.org/pdf/2407.02329?). Boundary integrity is tracked via the **Seam Discontinuity Index (`SDI`)**, defined as the ratio of horizontal and vertical Sobel gradient magnitudes along known tile boundaries (\(x, y \equiv 0 \pmod{\text{stride}}\)) relative to internal tile regions (\(1.00 = \text{perfectly seamless}\)). Alongside image quality metrics, the harness logs systems telemetry—including peak GPU VRAM allocated (`torch.cuda.max_memory_allocated()`), peak reserved VRAM, peak system RAM RSS, wall-clock latency per stage, and bitwise reproducibility across interrupted/resumed runs—and concludes with a structured 2-Alternative Forced Choice (2AFC) blind human evaluation protocol (modeled on SyncDiffusion's user study design[11](https://arxiv.org/html/2306.05178v3)) assessing Global Coherence, Regional Prompt Adherence, and Local Sharpness.

[Interactive visualization 3 (frame_source_saved)](https://1oebmi6ro9u63kqwq8ugckjvdnbg158bvz91x1bz3enqm6ijdb-h998661601.scf.usercontent.goog/gemini-code-immersive/shim.html?origin=https%3A%2F%2Fgemini.google.com&cache=1)

## Execution Roadmap and Sequenced Engineering Recommendations

Synthesizing the findings across all ten research tracks confirms that every foundational question is resolved with high empirical confidence. To transition from this completed research stage into architectural design and phased implementation without regression risk, development is structured into four sequential engineering phases.

In **Phase 1 (Core Extrapolation & Low-VRAM Streaming Engine — Tracks 1–4)**, engineering focuses on constructing the pinned-memory CPU Master Latent Canvas and the `ModelPatcher` `unet_function_wrapper` tile streamer supporting configurable `view_batch_size` (\(1\text{–}16\)) and automatic OOM step-down recovery[10](https://note.com/198619891990/n/nf8d0cb57d57e). This phase integrates **HiDiffusion (`RAU-Net` + `MSW-MSA`)** for SDXL (vendored under Apache-2.0 from `sd-forge-hidiffusion`)[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf) alongside **Adaptively Rectified / Time-Aware 3D RoPE** coordinate offsetting for Anima 2B and 3.8B v2[15](https://ostris.com/docs/ai-toolkit/models/anima). Concurrently, the **AccDiffusion v2** cross-attention binarizer (\(B_{i,j}\)) is wired into the \(1024 \times 1024\) anchor pass alongside position-wise bijective dilated sampling, **FreeScale / RectifiedHR** detail fusion, and Forge Neo's **Tiled VAE + DeGrid** post-processing[2](https://github.com/Haoming02/sd-webui-forge-classic).

In **Phase 2 (Regional Conditioning, Masked On-the-Fly LoRAs, and Structural Guidance — Tracks 5 & 8)**, development establishes the sequential regional text encoder pipeline with its SHA-256 CPU conditioning cache, handling dual CLIP for SDXL and sequential `qwen_3_06b_base` + `qwen35_4b` + timestep-aware `Semantic Connector v2` for Anima 3.8B[6](https://github.com/GumGum10/forge-anima-3.8B). This phase implements the Three-Tier Regional LoRA isolation engine—comprising `Tier 1` in-place spatially masked LoRA deltas (\(\Delta W_r x \odot M_r\)) with MIGC++-inspired `AttentionCouple`, `Tier 2` `PyTorch SDPA` block-diagonal `attn1` masking, and `Tier 3` tile-aligned LoRA pruning[2](https://github.com/Haoming02/sd-webui-forge-classic)—and wires automatic in-memory rasterization of regional masks into Forge Neo's native **Anima Region ControlNet (`LLLite`)** at `end_percent = 0.45`[2](https://github.com/Haoming02/sd-webui-forge-classic).

In **Phase 3 (Multilayer Konva.js Regional Canvas and CRF v1 Protocol — Tracks 6 & 7)**, frontend work delivers the multi-canvas **Konva.js** editor inside Gradio 4.40.0 featuring persistent multilayer visibility, non-destructive `bbox`, `polygon`, and `brush` editing, real-time Effective Conditioning Heatmap visualization, and bi-directional binding to regional prompt and LoRA inspector cards[2](https://github.com/Haoming02/sd-webui-forge-classic). Simultaneously, the **CRF v1** JSON Schema validator, clipboard/file `.crf.zip` importer/exporter, aspect-ratio mismatch resolver, and PNG `infotext` round-trip serializer are integrated[6](https://github.com/GumGum10/forge-anima-3.8B).

Finally, **Phase 4 (Ecosystem Integration and Automated Benchmark Harness — Tracks 9 & 10)** completes cooperative hook chaining and `postprocess()` teardown for `sd-forge-couple`, `ControlNet`, `ADetailer`, `sd-dynamic-prompts`, and `NegPiP`[44](https://github.com/Haoming02/sd-forge-couple), and executes the four-category **BM-01–BM-04** automated benchmark suite across 4 GB, 6 GB, 8 GB, and 12 GB memory profiles to lock in default hyperparameters prior to public release.

## Interactive preview inventory

- [Visualization 1](https://24fnp0fbhvl5vu3ysxovghygfht8tr25g7isn9aaidf2zcbwc0-h998661601.scf.usercontent.goog/gemini-code-immersive/shim.html?origin=https%3A%2F%2Fgemini.google.com&cache=1) — frame_source_saved
- [Visualization 2](#) — cross_origin_reference_only
- [Visualization 3](https://1oebmi6ro9u63kqwq8ugckjvdnbg158bvz91x1bz3enqm6ijdb-h998661601.scf.usercontent.goog/gemini-code-immersive/shim.html?origin=https%3A%2F%2Fgemini.google.com&cache=1) — frame_source_saved