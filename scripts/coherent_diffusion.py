"""Forge Neo Coherent Diffusion tab and noninvasive txt2img diagnostics.

P07 UI invocation is opt-in. Native Forge sampler, checkpoint loader, and
existing Anima Q4 encoder are never replaced by registering this extension.
"""
import gradio as gr
import json
import io
import base64
import uuid
import os
from pathlib import Path
from html import escape
from modules import scripts, script_callbacks
from coherent_diffusion.forge_bridge import inspect_loaded_host, inspect_process
from coherent_diffusion.forge_bridge.entrypoint import gradio_generate, cancel_generation
from coherent_diffusion.forge_bridge.diagnostics import (
    install_gradio_probe, record as debug_record, recent_text as debug_recent,
    status_text as debug_status)
install_gradio_probe()

_LAST = None
_EDITOR_PATH = Path(__file__).resolve().parents[1] / "coherent_diffusion" / "editor" / "index.html"

# The local srcdoc iframe transfers validated CRF to same-origin Gradio
# fields. Nested Gradio Blocks js= hooks are not reliable when rendered in Forge.

def editor_iframe():
    """No network resources, CDN dependency or startup sampler hooks."""
    page = _EDITOR_PATH.read_text(encoding="utf-8")
    return ('<iframe title="Coherent Diffusion region editor" '
            'referrerpolicy="no-referrer" loading="lazy" '
            'style="width:100%;height:890px;min-height:650px;border:1px solid #303748;'
            'border-radius:10px;background:#0c0f15;" '
            f'srcdoc="{escape(page, quote=True)}"></iframe>')


_EDITOR_FIELDS = ("family", "mode", "width", "height", "tile", "overlap",
                  "halo", "steps", "cfg", "seed", "cpu_mib", "window_mib")
_EDITOR_ARG_INDEX = (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)


def _resolve_canvas_generation(inputs):
    """Atomic one-shot editor handoff, independent of Gradio dropdown DOM.

    The CRF remains a pure composition document; UI execution settings are
    validated separately, cannot mutate LoRA opt-in, and are not retained by
    subsequent native-form submissions.
    """
    if len(inputs) != 18:
        raise ValueError("Expected 17 native Forge inputs and one canvas handoff")
    payload = inputs[-1]
    effective = list(inputs[:-1])
    if not payload:
        return effective
    if not isinstance(payload, str) or len(payload) > 4096:
        raise ValueError("Canvas generation handoff is malformed or oversized")
    try:
        envelope = json.loads(payload)
    except (ValueError, TypeError) as exc:
        raise ValueError("Canvas generation settings are not valid JSON") from exc
    if (not isinstance(envelope, dict) or
            set(envelope) != {"format", "version", "settings"} or
            envelope["format"] != "coherent-diffusion-generation" or
            type(envelope["version"]) is not int or envelope["version"] != 1):
        raise ValueError("Unsupported canvas generation settings envelope")
    settings = envelope["settings"]
    if not isinstance(settings, dict) or set(settings) != set(_EDITOR_FIELDS):
        raise ValueError("Canvas settings keys do not match the verified Forge form")
    for name, index in zip(_EDITOR_FIELDS, _EDITOR_ARG_INDEX):
        value = settings[name]
        if name in ("family", "mode"):
            if not isinstance(value, str):
                raise ValueError("Invalid canvas "+name)
        elif name == "cfg":
            if type(value) not in (float, int):
                raise ValueError("Invalid canvas CFG")
        elif type(value) is not int:
            raise ValueError("Invalid integer canvas parameter: "+name)
        effective[index] = value
    return effective


def _generate_for_ui(*inputs):
    """Persist the lossless full PNG and embed only a bounded-size preview.

    Never discard the original just because the output exceeds the old
    64-MiB HTML data-URI limit. Gradio versions differ in FileData plumbing.
    """
    debug_record("ui_generate_submit")
    try:
        effective = _resolve_canvas_generation(inputs)
        image, evidence = gradio_generate(*effective)
    except (ValueError, TypeError) as exc:
        # Clear the one-shot form bridge even when validation fails; an
        # invalid editor payload must not contaminate later native submits.
        message = str(exc)[:160]
        return ("<p role='status'>Invalid canvas generation settings.</p>",
                json.dumps({"status":"error","type":"InvalidEditorSettings",
                            "detail":message,"partial_image":False}), "")
    if image is None:
        debug_record("ui_generation_failed", type=evidence.get("type"),
                     phase=evidence.get("phase"))
        return "<p role='status'>No image produced. See execution evidence below.</p>", json.dumps(evidence, ensure_ascii=False, indent=2), ""
    from modules import paths
    outdir = Path(paths.data_path).resolve() / "outputs" / "coherent-diffusion"
    outdir.mkdir(parents=True, exist_ok=True)
    filename = f"coherent-diffusion-{uuid.uuid4().hex}.png"
    destination = outdir / filename
    temporary = outdir / (filename + ".part")
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    preview = image.copy()
    preview.thumbnail((1600, 1600))
    buffer = io.BytesIO()
    preview.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    debug_record("ui_png_saved", width=image.width, height=image.height)
    evidence = {**evidence, "saved_png": str(destination),
                "preview_width": preview.width, "preview_height": preview.height}
    report = json.dumps(evidence, ensure_ascii=False, indent=2)
    return (f'<div><img alt="Coherent Diffusion generated preview" '
            f'style="display:block;max-width:100%;max-height:820px;margin:auto" '
            f'src="data:image/png;base64,{encoded}">'
            f'<p>Lossless full-size PNG saved to Forge outputs/coherent-diffusion/ '
            f'({image.width} × {image.height}).</p></div>'), report, ""


def _debug_poll():
    return debug_recent(), debug_status()


def editor_tab():
    """Full-width editor and a scoped Gradio 4 generation surface."""
    with gr.Blocks(analytics_enabled=False) as panel:
        gr.HTML(value=editor_iframe())
        with gr.Accordion("Generate · loaded Forge checkpoint", open=True):
            gr.Markdown(
                "Generate uses Forge-native loading for your selected checkpoint/encoder. It does not "
                "change the native txt2img sampler. Click **Generate image** "
                "in the canvas to transfer its CRF, or use these controls directly. "
                "**Unverified installed-host compatibility:** start with a small scene. "
                "Optional controls, real-host regional LoRA isolation, batch, Hires Fix and img2img "
                "are not activated without live integration evidence.")
            with gr.Row():
                model = gr.Dropdown(["sdxl", "anima"], value="sdxl", label="Loaded model family", elem_id="cd_model")
                mode = gr.Dropdown(["progressive", "direct"], value="progressive", label="Coherent trajectory", elem_id="cd_mode")
                width = gr.Number(label="Width (px)", value=1024, precision=0, elem_id="cd_width")
                height = gr.Number(label="Height (px)", value=1024, precision=0, elem_id="cd_height")
            with gr.Row():
                prompt = gr.Textbox(label="Scene prompt (used if CRF input empty)",
                                    placeholder="One mountain, lake and coherent evening light", scale=3)
                negative = gr.Textbox(label="Negative prompt", scale=2)
            with gr.Row():
                tile = gr.Number(label="Tile (px)", value=512, precision=0, elem_id="cd_tile")
                overlap = gr.Number(label="Overlap (px)", value=128, precision=0, elem_id="cd_overlap")
                halo = gr.Number(label="Halo (px)", value=64, precision=0, elem_id="cd_halo")
                steps = gr.Number(label="Euler/flow steps", value=20, precision=0, elem_id="cd_steps")
                cfg = gr.Number(label="CFG", value=4.5, elem_id="cd_cfg")
                seed = gr.Number(label="Seed", value=42, precision=0, elem_id="cd_seed")
            with gr.Accordion("CRF import and conservative memory budgets", open=False):
                scene = gr.Textbox(label="Validated CRF JSON (optional)", lines=5, elem_id="cd_scene_json")
                archive = gr.Textbox(label="Verified CRF ZIP transfer (automatically filled by editor)",
                                     lines=1, elem_id="cd_archive_b64")
                editor_params = gr.Textbox(
                    label="Canvas generation settings handoff (one-shot, cleared after Generate)",
                    value="", lines=1, elem_id="cd_editor_settings")
                with gr.Row():
                    cpu_mib = gr.Number(label="Maximum CPU workspace (MiB)", value=4096, precision=0, elem_id="cd_cpu_mib")
                    window_mib = gr.Number(label="Maximum latent window (MiB)", value=512, precision=0, elem_id="cd_window_mib")
                activate_loras = gr.Checkbox(
                    value=False, elem_id="cd_activate_loras",
                    label="Activate regional denoiser LoRAs (experimental, explicit opt-in)")
                gr.Markdown(
                    "LoRA references are preserved in your saved CRF. By default they "
                    "are **bypassed for this run**, and their number is reported in "
                    "execution evidence. Enable only for real locally registered "
                    "denoiser LoRAs; regional text-encoder/Q4 LoRAs remain unsupported.")
            with gr.Row():
                submit = gr.Button("Generate with loaded Forge model", variant="primary", elem_id="cd_submit")
                stop = gr.Button("Stop coherent generation", elem_id="cd_stop")
                cancelled = gr.Textbox(label="Cancellation", interactive=False, scale=2, elem_id="cd_cancel_state")
            with gr.Row():
                output = gr.HTML(label="Generated image", value="<p>No generation submitted.</p>", elem_id="cd_image_result")
                result = gr.Textbox(label="Execution evidence / errors", lines=10, interactive=False, elem_id="cd_execution_evidence")
            job = submit.click(
                fn=_generate_for_ui,
                inputs=[scene, archive, prompt, negative, model, mode, width, height,
                        tile, overlap, halo, steps, cfg, seed, cpu_mib, window_mib,
                        activate_loras, editor_params],
                outputs=[output, result, editor_params],
                queue=True, show_progress="full")
            stop.click(fn=cancel_generation, inputs=[], outputs=[cancelled], queue=False)
        with gr.Accordion("Live diagnostics — persistent redacted events", open=True):
            gr.Markdown(
                "Auto-updating server events appear below. Use the always-visible "
                "**CD live debug → Download logs** button at the bottom right "
                "of Forge to export these together with browser/Gradio event IDs. "
                "Raw prompts, CRF documents, tokens and request payloads are not recorded. "
                "The server journal is retained in Forge outputs/coherent-diffusion/diagnostics/.")
            live_status = gr.Textbox(
                label="Live generation progress — actual prediction evaluations (upper bound, not ETA)",
                lines=2, interactive=False, elem_id="cd_debug_status",
                value=debug_status())
            debug_log = gr.Textbox(
                label="Live Python / Gradio events (JSON lines)", lines=11,
                interactive=False, elem_id="cd_debug_backend",
                value=debug_recent())
            if hasattr(gr, "Timer"):
                debug_clock = gr.Timer(value=2.0, active=True)
                debug_clock.tick(fn=_debug_poll, inputs=[],
                                 outputs=[debug_log, live_status],
                                 queue=False, show_progress="hidden")
            else:
                gr.Button("Refresh diagnostic log").click(
                    fn=_debug_poll, inputs=[],
                    outputs=[debug_log, live_status], queue=False)
    return [(panel, "Coherent Diffusion", "coherent_diffusion")]


script_callbacks.on_ui_tabs(editor_tab)


def probe_ui():
    return _LAST if _LAST is not None else inspect_loaded_host()


class Script(scripts.Script):
    def title(self):
        return "Coherent Diffusion (editor, opt-in generation, diagnostics)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if not is_img2img else False

    def ui(self, is_img2img):
        with gr.Accordion("Coherent Diffusion — diagnostics", open=False):
            gr.Markdown("Read-only loaded-model inspection. Native txt2img remains untouched.")
            enabled = gr.Checkbox(label="Capture read-only metadata during txt2img", value=False)
            inspect_button = gr.Button("Inspect loaded model", elem_id="cd_probe_button")
            result = gr.JSON(label="Capability / encoder evidence", value={}, elem_id="cd_probe_evidence")
            inspect_button.click(fn=probe_ui, inputs=[], outputs=[result], queue=False)
        gr.Markdown("Open the **Coherent Diffusion** full-width tab for scene editing and opt-in generation.")
        return [enabled]

    def process(self, p, enabled=False):
        global _LAST
        if enabled:
            try:
                _LAST = inspect_process(p)
            except Exception as exc:
                _LAST = {"probe_error": f"{type(exc).__name__}: {exc}",
                         "generation_affected": False}
