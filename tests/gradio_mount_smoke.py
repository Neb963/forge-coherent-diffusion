"""Miniature real Gradio 4 host exercising the actual Forge Script.ui mount.

This does not simulate Forge model internals or prove extension discovery.
"""
import importlib.util
import sys
import threading
import types
import tempfile
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

mod=types.ModuleType("modules")
scripts=types.ModuleType("modules.scripts")
scripts.Script=type("Script",(object,),{})
scripts.AlwaysVisible=object()
mod.scripts=scripts
callbacks=types.ModuleType("modules.script_callbacks")
callbacks.handlers=[]
callbacks.on_ui_tabs=lambda fn:callbacks.handlers.append(fn)
mod.script_callbacks=callbacks
paths=types.ModuleType("modules.paths")
paths.data_path=tempfile.mkdtemp(prefix="coherent-gradio-smoke-")
mod.paths=paths
sys.modules["modules.paths"]=paths
sys.modules["modules.script_callbacks"]=callbacks
sys.modules["modules"]=mod
sys.modules["modules.scripts"]=scripts

spec=importlib.util.spec_from_file_location(
    "cd_gradio_integration",Path(__file__).resolve().parents[1]/"scripts"/"coherent_diffusion.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# These callbacks are deliberately synthetic. Browser tests below verify
# *transport and handler ownership*, never that Forge GPU inference works.
def isolated_generate(*inputs):
    from PIL import Image
    preview=Image.new("RGB",(640,384),"navy") if inputs[13] == 777 else None
    return preview, {"status": "success" if preview else "error",
                     "type": "SMOKE_BACKEND_CALLED",
                  "detail": "Gradio Python Generate callback was invoked",
                  "input_count": len(inputs), "partial_image": False,
                  "family": inputs[4], "width": inputs[6], "height": inputs[7],
                  "tile": inputs[8], "overlap": inputs[9], "halo": inputs[10],
                  "steps": inputs[11], "cfg": inputs[12], "seed": inputs[13],
                  "cpu_mib": inputs[14], "window_mib": inputs[15]}
module.gradio_generate = isolated_generate
def probe_ui():
    return {"status": "SMOKE_INSPECT_BACKEND_CALLED"}
module.probe_ui = probe_ui

editor_tabs=[entry for maker in callbacks.handlers for entry in maker()]
with gr.Blocks(title="P02 Gradio 4 mount smoke") as demo:
    # Forge has frontend-only listeners with fn=None mixed among Python
    # handlers. They are legal unless the browser mistakenly queues them.
    with gr.Tab("txt2img"):
        host_only = gr.Button("Forge frontend-only mock", elem_id="cd_host_js_only")
        host_only.click(fn=None, js="() => console.debug('host frontend action')")
        module.Script().ui(False)
    for tab,title,elem_id in editor_tabs:
        with gr.Tab(title,elem_id="tab_"+elem_id):
            tab.render()
# Gradio 4.x must copy the nested tab's callable functions into the root
# event map. Missing fn produces the exact user traceback inside call_function.
backend_functions = {"_generate_for_ui", "cancel_generation", "probe_ui", "_debug_poll"}
registered = [(key, getattr(f, "fn", None)) for key, f in demo.fns.items()
              if getattr(getattr(f, "fn", None), "__name__", None) in backend_functions]
assert len(registered) == 4, (
    "Forge-like nested Blocks rendered without all four Python handlers: "
    + repr([(i, getattr(fn, "__name__", None)) for i, fn in registered]))
config = demo.get_config_file()
deps = config.get("dependencies", [])
assert any(d.get("backend_fn") is False for d in deps), (
    "Host-like frontend-only Gradio callback not represented")
for index, handler in registered:
    matches = [d for d in deps if d.get("id") == index]
    assert matches, f"Registered handler {index} missing from root config"
    assert matches[0].get("backend_fn") is True, (
        f"Callable Gradio handler {index} incorrectly rendered as no backend: "
        + repr(matches[0]))
print("P07_BACKEND_FN_MAP", [(i, f.__name__) for i, f in registered], flush=True)

demo.launch(server_name="127.0.0.1",server_port=18746,
            show_error=True,prevent_thread_lock=True,quiet=True)
print("P02_GRADIO_READY",flush=True)
threading.Event().wait()
