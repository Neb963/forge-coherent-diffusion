"""Forge Neo P01 diagnostics. No generation override, hooks or backend patches."""
import gradio as gr
from modules import scripts
from coherent_diffusion.forge_bridge import inspect_loaded_host, inspect_process

_LAST = None

def probe_ui():
    return _LAST if _LAST is not None else inspect_loaded_host()

class Script(scripts.Script):
    def title(self):
        return "Coherent Diffusion (P01 diagnostics)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if not is_img2img else False

    def ui(self, is_img2img):
        with gr.Accordion("Coherent Diffusion — diagnostics only", open=False):
            gr.Markdown("P01 foundation: **generation and regional editor are not installed yet**. Inspect the already-loaded model without changing Forge's pipeline. Q4 and positional support require later live proof.")
            enabled = gr.Checkbox(label="Capture read-only metadata during txt2img", value=False)
            inspect_button = gr.Button("Inspect loaded model")
            result = gr.JSON(label="Capability / encoder evidence", value={})
            inspect_button.click(fn=probe_ui, inputs=[], outputs=[result], queue=False)
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
