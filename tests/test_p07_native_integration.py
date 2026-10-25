"""P07 native job ownership/loading and safe CRF LoRA execution contracts.

Offline stubs prove call ordering and state isolation, not real Forge GPU output.
"""
import hashlib
import json
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from coherent_diffusion.crf import canonical_json
from coherent_diffusion.forge_bridge import entrypoint as ep


def args():
    return ("", "", "coastal landscape", "", "anima", "direct", 512, 512,
            512, 128, 64, 2, 1.0, 42, 1024, 128)


class NativeQueueLifecycle(unittest.TestCase):
    def run_fake(self, *, fail=False):
        events = []
        lock = threading.Lock()
        pristine = types.SimpleNamespace(unet=object(), clip=object())
        class Holder:
            def shallow_copy(self):
                events.append("copy_pristine")
                return types.SimpleNamespace(unet=pristine.unet, clip=pristine.clip)
        original = object()
        host = types.SimpleNamespace(forge_objects=original,
                                     forge_objects_original=Holder())
        native = types.ModuleType("modules")
        sd_models = types.ModuleType("modules.sd_models")
        sd_models.model_data = types.SimpleNamespace(sd_model=object())
        def reload():
            self.assertTrue(lock.locked())
            events.append("forge_reload")
            return host, False
        sd_models.forge_model_reload = reload
        shared = types.ModuleType("modules.shared")
        state = types.SimpleNamespace(interrupted=False)
        def begin(job):
            self.assertTrue(lock.locked())
            events.append(("begin", job))
        def end():
            self.assertTrue(lock.locked())
            events.append("end")
        state.begin, state.end = begin, end
        shared.state = state
        call_queue = types.ModuleType("modules.call_queue")
        call_queue.queue_lock = lock
        main_entry = types.ModuleType("modules_forge.main_entry")
        main_entry.refresh_model_loading_parameters = lambda: events.append("refresh")
        forge = types.ModuleType("modules_forge")
        forge.main_entry = main_entry
        native.sd_models, native.shared, native.call_queue = sd_models, shared, call_queue
        def run(plan, host, cancelled=None):
            self.assertTrue(lock.locked())
            self.assertIsNot(host.forge_objects, original)
            self.assertIs(host.forge_objects.clip, pristine.clip)
            events.append("execute")
            if fail:
                raise RuntimeError("simulated sampling failure")
            return "image", {"ok": True}
        with patch.dict(sys.modules, {"modules": native, "modules.sd_models": sd_models,
                                      "modules.shared": shared, "modules.call_queue": call_queue,
                                      "modules_forge": forge, "modules_forge.main_entry": main_entry}), \
             patch.object(ep, "preflight", return_value="plan") as preflight, \
             patch.object(ep, "execute", side_effect=run):
            result = ep.gradio_generate(*args())
        self.assertIs(host.forge_objects, original)
        self.assertFalse(lock.locked())
        self.assertEqual(events, [("begin", "coherent-diffusion"),
                                  "refresh", "forge_reload", "copy_pristine", "execute", "end"])
        self.assertFalse(preflight.call_args.kwargs["activate_region_loras"])
        return result

    def test_native_loader_and_original_model_restore(self):
        self.assertEqual(self.run_fake(), ("image", {"ok": True}))

    def test_failure_does_not_leak_patchers_or_lock(self):
        image, result = self.run_fake(fail=True)
        self.assertIsNone(image)
        self.assertEqual(result["type"], "RuntimeError")
        self.assertFalse(result["partial_image"])


class UnetSourceContracts(unittest.TestCase):
    @unittest.skipIf(torch is None, "Needs torch to import SDXL backend")
    def test_default_sdxl_host_has_empty_container_not_active_hook(self):
        from coherent_diffusion.forge_bridge.sdxl import ForgeSDXL, UnsupportedSDXL
        class StableDiffusionXL:
            is_sdxl = True
            is_inpaint = False
            use_shift = False
        host = StableDiffusionXL()
        model = types.SimpleNamespace(apply_model=lambda *a: None)
        unet = types.SimpleNamespace(
            controlnet_linked_list=None, extra_concat_condition=None,
            model_options={"transformer_options": {}}, model=model)
        vae = types.SimpleNamespace(decode_tiled=lambda *a: None,
                                    upscale_ratio=8, latent_channels=4)
        host.forge_objects = types.SimpleNamespace(unet=unet, vae=vae)
        host.get_learned_conditioning = lambda *a: {}
        ForgeSDXL._check_host(host)
        unet.model_options["transformer_options"]["patches"] = [object()]
        with self.assertRaisesRegex(UnsupportedSDXL, "Unnegotiated"):
            ForgeSDXL._check_host(host)

    def test_starter_template_does_not_insert_fake_lora(self):
        ui = (Path(__file__).resolve().parents[1] /
              "coherent_diffusion/editor/index.html").read_text()
        self.assertNotIn("example_local_texture_lora", ui)
        self.assertIn("regional LoRA reference(s)", ui)


@unittest.skipIf(torch is None, "P07 tensor CI runs regional compiler tests")
class PerRunLoRAIntent(unittest.TestCase):
    def test_bypass_is_explicit_and_authored_hash_is_stable(self):
        from coherent_diffusion.forge_bridge.anima import ForgeAnima
        import torch
        doc = ep._default_doc("One landscape", "", 512, 512)
        doc["regions"] = [{
            "id": "sand", "name": "Sand", "geometry": {"type": "bbox", "bbox": [0, .5, 1, .5]},
            "prompt": "shore", "blend": {"weight": 1, "feather_px": 0, "z_index": 1},
            "loras": [{"id": "demo", "name": "example_local_texture_lora",
                        "weight": .6, "scope": "regional",
                        "requested_isolation": "approximate"}],
        }]
        text = canonical_json(doc)
        from coherent_diffusion.crf import parse_validate
        parse_validate(text)
        expected = hashlib.sha256(text.encode()).hexdigest()
        kwargs = dict(host=object(), family="anima", mode="direct", width=512, height=512,
                      tile=512, overlap=128, halo=64, steps=2, cfg=1,
                      seed=42, cpu_mib=1024, window_mib=128, scene_json=text)
        with patch.object(ForgeAnima, "check_host", return_value=2), \
             patch.object(ep, "_model_sigmas", return_value=(1.0, .5, 0.0)):
            p = ep.preflight(**kwargs)
            self.assertEqual(p.bypassed_region_loras, 1)
            self.assertFalse(p.activate_region_loras)
            self.assertEqual(p.document_hash, expected)
            self.assertEqual(p.document["regions"][0]["loras"], [])
            self.assertEqual(doc["regions"][0]["loras"][0]["name"], "example_local_texture_lora")
            active = ep.preflight(**kwargs, activate_region_loras=True)
            self.assertTrue(active.activate_region_loras)
            self.assertEqual(active.bypassed_region_loras, 0)
            self.assertEqual(active.document["regions"][0]["loras"][0]["name"],
                             "example_local_texture_lora")


if __name__ == "__main__":
    unittest.main()
