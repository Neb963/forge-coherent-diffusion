"""Forge Neo P01 diagnostics. No generation override, hooks or backend patches."""
import gradio as gr
from pathlib import Path
from html import escape
from modules import scripts
from coherent_diffusion.forge_bridge import inspect_loaded_host, inspect_process

_LAST = None

_EDITOR_PATH = Path(__file__).resolve().parents[1] / "coherent_diffusion" / "editor" / "index.html"

def editor_iframe():
    """Bundle the offline editor inline; no CDN, static routes or backend hooks."""
    page = _EDITOR_PATH.read_text(encoding="utf-8")
    return ('<iframe title="Coherent Diffusion region editor" '
            'referrerpolicy="no-referrer" loading="lazy" '
            'style="width:100%;height:890px;min-height:650px;border:1px solid #303748;'
            'border-radius:10px;background:#0c0f15;" '
            f'srcdoc="{escape(page, quote=True)}"></iframe>')

def probe_ui():
    return _LAST if _LAST is not None else inspect_loaded_host()

class Script(scripts.Script):
    def title(self):
        return "Coherent Diffusion (P02 editor + P01 diagnostics)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if not is_img2img else False

    def ui(self, is_img2img):
        with gr.Accordion("Coherent Diffusion — diagnostics only", open=False):
            gr.Markdown("P01 diagnostics remain read-only; **P02 editor is available below, while generation is not implemented**. Inspect the already-loaded model without changing Forge's pipeline. Q4 and positional support require later live proof.")
            enabled = gr.Checkbox(label="Capture read-only metadata during txt2img", value=False)
            inspect_button = gr.Button("Inspect loaded model")
            result = gr.JSON(label="Capability / encoder evidence", value={})
            inspect_button.click(fn=probe_ui, inputs=[], outputs=[result], queue=False)
        with gr.Accordion("Coherent Diffusion — regional composition editor (P02)", open=False):
            gr.Markdown("Local CRF editor; **generation remains unavailable** until sampler integration. Editor drafts stay in this browser.")
            gr.HTML(value=editor_iframe())
        return [enabled]

    def process(self, p, enabled=False):
        global _LAST
        if enabled:
            try:
                _LAST = inspect_process(p)
            except Exception as exc:
                # Diagnostics must never obstruct ordinary Forge generations.
                _LAST = {"probe_error": f"{type(exc).__name__}: {exc}",
                         "generation_affected": False}
