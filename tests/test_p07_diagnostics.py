"""P07 privacy-safe persistent diagnostics: real files and synthetic Gradio dispatcher."""
import asyncio
import json
import tempfile
import types
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from coherent_diffusion.forge_bridge import diagnostics as d


class DiagnosticsJournal(unittest.TestCase):
    def test_rotating_log_and_allowlisted_fields(self):
        with tempfile.TemporaryDirectory() as t, patch.object(d, "_data_dir", return_value=Path(t)):
            d.record("generate_trace", type="MemoryError", phase="preflight",
                     width=4096, height=4096,
                     prompt="SECRET_DO_NOT_SAVE", token="SECRET_DO_NOT_SAVE")
            path=d.diagnostic_file()
            self.assertTrue(path.is_file())
            first=json.loads(d.recent_text(1))
            self.assertEqual(first["width"],4096)
            self.assertNotIn("prompt",first)
            self.assertNotIn("token",first)
            self.assertNotIn("SECRET_DO_NOT_SAVE", path.read_text())
            with patch.object(d,"_MAX_FILE_BYTES",400):
                for _ in range(8):
                    d.record("repeat_diagnostic", phase="native_model_load",type="MemoryError")
            self.assertTrue(path.with_suffix(".jsonl.1").exists())
            self.assertLessEqual(path.stat().st_size,650)

    def test_authored_crf_hash_is_visible_without_prompt_text(self):
        digest = "9f" * 32
        with tempfile.TemporaryDirectory() as root, patch.object(
                d, "_data_dir", return_value=Path(root)):
            d.record("tile_plan", family="anima", width=2048, height=1536,
                     tile=1024, window_total=6, crf_sha256=digest,
                     prompt="A unique secret prompt", scene_json='{"private":1}')
            saved = json.loads(d.recent_text(1))
            self.assertEqual(saved["crf_sha256"], digest)
            self.assertEqual(saved["window_total"], 6)
            self.assertNotIn("prompt", saved)
            self.assertNotIn("scene_json", saved)
            self.assertNotIn("unique secret", d.diagnostic_file().read_text())

    def test_strict_value_redaction(self):
        with tempfile.TemporaryDirectory() as t, patch.object(d,"_data_dir",return_value=Path(t)):
            d.record("failure", type="RuntimeError", route="/queue/join",
                     api_name="bad prompt with spaces and token value",
                     targets=["42:click", "secret api-key with spaces"])
            lines=json.loads(d.recent_text(1))
            self.assertEqual(lines["type"],"RuntimeError")
            self.assertEqual(lines["api_name"],"[redacted]")
            self.assertEqual(lines["targets"][0],"42:click")
            self.assertEqual(lines["targets"][1],"[redacted]")

    def test_gradio_missing_callback_is_observed_not_swallowed(self):
        class BlockFn:
            fn=None
        block=BlockFn()
        class Blocks:
            async def call_function(self, block_fn, *args, **kwargs):
                if block_fn.fn is None:
                    raise IndexError("function has no backend method.")
                return "ok"
            def __init__(self):
                self.fns={8:block}
            def get_config_file(self):
                return {"dependencies":[{
                    "id":8, "backend_fn":False, "api_name":None,
                    "targets":[[123,"click"]]}]}
        stub=types.ModuleType("gradio")
        stub.Blocks=Blocks
        stub.__version__="4.40.0"
        with tempfile.TemporaryDirectory() as t, \
             patch.object(d,"_data_dir",return_value=Path(t)), \
             patch.dict(sys.modules,{"gradio":stub}):
            self.assertTrue(d.install_gradio_probe())
            with self.assertRaisesRegex(IndexError,"function has no backend method"):
                asyncio.run(Blocks().call_function(block, []))
            row=json.loads(d.recent_text(1))
            self.assertEqual(row["event"],"gradio_missing_backend")
            self.assertEqual(row["fn_index"],8)
            self.assertEqual(row["backend_fn"],False)
            self.assertEqual(row["targets"],["123:click"])
            self.assertTrue(d.install_gradio_probe())
            self.assertEqual(asyncio.run(Blocks().call_function(
                types.SimpleNamespace(fn=True), [])),"ok")


if __name__=="__main__":
    unittest.main()
