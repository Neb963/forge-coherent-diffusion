"""P07 atomic HTML canvas -> Gradio Python configuration contract.

When Gradio Svelte dropdowns lag React-style DOM events, editor inputs must
still control the *actual* model family, sigma schedule, tile/overlap, and CFG.
The handoff is deliberately one-shot and never writes settings into CRF.
"""
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class AtomicEditorSettings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module=types.ModuleType("modules")
        scripts=types.ModuleType("modules.scripts")
        scripts.Script=type("Script",(),{})
        scripts.AlwaysVisible=True
        callbacks=types.ModuleType("modules.script_callbacks")
        callbacks.on_ui_tabs=lambda fn: None
        module.scripts=scripts
        module.script_callbacks=callbacks
        gr=types.ModuleType("gradio")
        src=Path(__file__).resolve().parents[1]/"scripts"/"coherent_diffusion.py"
        spec=importlib.util.spec_from_file_location("p07_editor_atomic_handoff",src)
        loaded=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{"modules":module,
                                     "modules.scripts":scripts,
                                     "modules.script_callbacks":callbacks,
                                     "gradio":gr}):
            spec.loader.exec_module(loaded)
        cls.bridge=loaded

    def parameters(self):
        base=["", "", "", "", "sdxl", "progressive",
              1024, 1024, 512, 128, 64, 20, 4.5, 42, 4096, 512, False]
        changed={"family":"anima","mode":"progressive",
                 "width":2048,"height":1536,"tile":512,
                 "overlap":192,"halo":128,"steps":9,"cfg":1.25,
                 "seed":666,"cpu_mib":2048,"window_mib":256}
        envelope=json.dumps({"format":"coherent-diffusion-generation",
                             "version":1,"settings":changed})
        return base+[envelope],changed

    def test_one_shot_authoritative_values(self):
        inputs,changed=self.parameters()
        actual=self.bridge._resolve_canvas_generation(inputs)
        self.assertEqual(len(actual),17)
        for name,index in zip(self.bridge._EDITOR_FIELDS,self.bridge._EDITOR_ARG_INDEX):
            self.assertEqual(actual[index],changed[name])
        self.assertEqual(actual[16],False,"LoRA opt-in stays Forge-controlled")
        native=self.bridge._resolve_canvas_generation(inputs[:-1]+[""])
        self.assertEqual(native[4],"sdxl")
        self.assertEqual(native[11],20)

    def test_rejects_bad_envelope_and_extra_fields(self):
        inputs,_=self.parameters()
        invalid=[
            '{"format":"bad"}','{invalid',
            json.dumps({"format":"coherent-diffusion-generation","version":1,
                        "settings":{"family":"anima"}}),
            json.dumps({"format":"coherent-diffusion-generation","version":1,
                        "settings":{**json.loads(inputs[-1])["settings"],"secret":"x"}}),
            json.dumps({"format":"coherent-diffusion-generation","version":1,
                        "settings":{**json.loads(inputs[-1])["settings"],"steps":True}})
        ]
        for bad in invalid:
            with self.subTest(bad=bad[:70]),self.assertRaises(ValueError):
                self.bridge._resolve_canvas_generation(inputs[:-1]+[bad])

    def test_output_clears_editor_override_even_on_failure(self):
        inputs,_=self.parameters()
        with patch.object(self.bridge,"gradio_generate",return_value=(
            None,{"status":"error","type":"GPUError","phase":"decode"})) as stub:
            html,report,clear=self.bridge._generate_for_ui(*inputs)
        self.assertEqual(clear,"")
        self.assertEqual(json.loads(report)["type"],"GPUError")
        self.assertEqual(stub.call_args.args[4],"anima")
        self.assertEqual(stub.call_args.args[11],9)
        with patch.object(self.bridge,"gradio_generate") as stub:
            html,report,clear=self.bridge._generate_for_ui(
                *(inputs[:-1]+["{broken"]))
        self.assertEqual(clear,"")
        self.assertEqual(json.loads(report)["type"],"InvalidEditorSettings")
        stub.assert_not_called()


if __name__=="__main__":
    unittest.main()
