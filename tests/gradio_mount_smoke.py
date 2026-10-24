"""Miniature real Gradio 4 host exercising the actual Forge Script.ui mount.

This does not simulate Forge model internals or prove extension discovery.
"""
import importlib.util
import sys
import threading
import types
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

mod=types.ModuleType("modules")
scripts=types.ModuleType("modules.scripts")
scripts.Script=type("Script",(object,),{})
scripts.AlwaysVisible=object()
mod.scripts=scripts
sys.modules["modules"]=mod
sys.modules["modules.scripts"]=scripts

spec=importlib.util.spec_from_file_location(
    "cd_gradio_integration",Path(__file__).resolve().parents[1]/"scripts"/"coherent_diffusion.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with gr.Blocks(title="P02 Gradio 4 mount smoke") as demo:
    with gr.Tab("txt2img"):
        module.Script().ui(False)
demo.launch(server_name="127.0.0.1",server_port=18746,
            show_error=True,prevent_thread_lock=True,quiet=True)
print("P02_GRADIO_READY",flush=True)
threading.Event().wait()
