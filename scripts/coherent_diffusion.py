"""Forge Neo Coherent Diffusion tab and noninvasive txt2img diagnostics.

P07 UI invocation is opt-in. Native Forge sampler, checkpoint loader, and
existing Anima Q4 encoder are never replaced by registering this extension.
"""
import gradio as gr
import json
from pathlib import Path
from html import escape
from modules import scripts, script_callbacks
from coherent_diffusion.forge_bridge import inspect_loaded_host, inspect_process
from coherent_diffusion.forge_bridge.entrypoint import gradio_generate, cancel_generation

_LAST = None
_EDITOR_PATH = Path(__file__).resolve().parents[1] / "coherent_diffusion" / "editor" / "index.html"

# Gradio 4 JS runs in the parent UI, not inside the iframe. Only accept messages
# from OUR srcdoc iframe, never arbitrary same-origin web messages. The editor
# retains all locally saved drafts; data transfer does not alter the document.
_BRIDGE_JS = r"""() => {
  if (window.__coherentDiffusionBridgeInstalled) return;
  window.__coherentDiffusionBridgeInstalled = true;
  function update(id, value) {
    const root = document.getElementById(id);
    const field = root?.querySelector('textarea,input,select');
    if (!field) throw new Error('Forge generation field unavailable: ' + id);
    const descriptor = Object.getOwnPropertyDescriptor(
      field.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype :
      field.tagName === 'SELECT' ? HTMLSelectElement.prototype :
      HTMLInputElement.prototype, 'value');
    descriptor.set.call(field, String(value));
    field.dispatchEvent(new Event('input', {bubbles:true}));
    field.dispatchEvent(new Event('change', {bubbles:true}));
  }
  window.addEventListener('message', event => {
    const frame = document.querySelector('iframe[title="Coherent Diffusion region editor"]');
    if (!frame || event.source !== frame.contentWindow) return;
    const data = event.data;
    if (!data || data.kind !== 'coherent-diffusion-generate-v1') return;
    try {
      if (typeof data.scene !== 'string' || typeof data.archive !== 'string' ||
          data.scene.length > 2_100_000 || data.archive.length > 34_000_000)
        throw new Error('Composition transfer exceeds size budget');
      const settings = data.settings || {};
      if (!['sdxl','anima'].includes(settings.model) ||
          !['direct','progressive'].includes(settings.strategy))
        throw new Error('Invalid generation mode');
      update('cd_scene_json', data.scene);
      update('cd_archive_b64', data.archive);
      update('cd_model', settings.model);
      update('cd_mode', settings.strategy);
      update('cd_width', settings.width);
      update('cd_height', settings.height);
      update('cd_tile', settings.tile === 'auto' ? 512 : settings.tile);
      const submit = document.querySelector('#cd_submit button');
      if (!submit) throw new Error('Forge Generate button missing');
      submit.click();
    } catch (error) {
      frame.contentWindow.postMessage(
        {kind:'coherent-diffusion-transfer-error-v1', detail:String(error.message)}, '*');
    }
  });
}"""


def editor_iframe():
    """No network resources, CDN dependency or startup sampler hooks."""
    page = _EDITOR_PATH.read_text(encoding="utf-8")
    return ('<iframe title="Coherent Diffusion region editor" '
            'referrerpolicy="no-referrer" loading="lazy" '
            'style="width:100%;height:890px;min-height:650px;border:1px solid #303748;'
            'border-radius:10px;background:#0c0f15;" '
            f'srcdoc="{escape(page, quote=True)}"></iframe>')


def _generate_for_ui(*inputs):
    image, evidence = gradio_generate(*inputs)
    return image, json.dumps(evidence, ensure_ascii=False, indent=2)


def editor_tab():
    """Full-width editor and a scoped Gradio 4 generation surface."""
    with gr.Blocks(analytics_enabled=False, js=_BRIDGE_JS) as panel:
        gr.HTML(value=editor_iframe())
        with gr.Accordion("Generate · loaded Forge checkpoint", open=True):
            gr.Markdown(
                "Generate uses the **currently loaded** SDXL or Anima model; it does not "
                "load checkpoints or change the native txt2img sampler. Click **Generate image** "
                "in the canvas to transfer its CRF, or use these controls directly. "
                "**Unverified installed-host compatibility:** start with a small scene. "
                "Optional controls, regional LoRA ownership, batch, Hires Fix and img2img "
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
                overlap = gr.Number(label="Overlap (px)", value=128, precision=0)
                halo = gr.Number(label="Halo (px)", value=64, precision=0)
                steps = gr.Number(label="Euler/flow steps", value=20, precision=0)
                cfg = gr.Number(label="CFG", value=4.5)
                seed = gr.Number(label="Seed", value=42, precision=0)
            with gr.Accordion("CRF import and conservative memory budgets", open=False):
                scene = gr.Textbox(label="Validated CRF JSON (optional)", lines=5, elem_id="cd_scene_json")
                archive = gr.Textbox(label="Verified CRF ZIP transfer (automatically filled by editor)",
                                     lines=1, elem_id="cd_archive_b64")
                with gr.Row():
                    cpu_mib = gr.Number(label="Maximum CPU workspace (MiB)", value=4096, precision=0)
                    window_mib = gr.Number(label="Maximum latent window (MiB)", value=512, precision=0)
            with gr.Row():
                submit = gr.Button("Generate with loaded Forge model", variant="primary", elem_id="cd_submit")
                stop = gr.Button("Stop coherent generation")
                cancelled = gr.Textbox(label="Cancellation", interactive=False, scale=2)
            with gr.Row():
                output = gr.Image(label="Generated image", type="pil", interactive=False)
                result = gr.Textbox(label="Execution evidence / errors", lines=10, interactive=False)
            job = submit.click(
                fn=_generate_for_ui,
                inputs=[scene, archive, prompt, negative, model, mode, width, height,
                        tile, overlap, halo, steps, cfg, seed, cpu_mib, window_mib],
                outputs=[output, result],
                queue=True, show_progress="full")
            stop.click(fn=cancel_generation, inputs=[], outputs=[cancelled], queue=False)
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
            inspect_button = gr.Button("Inspect loaded model")
            result = gr.JSON(label="Capability / encoder evidence", value={})
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
