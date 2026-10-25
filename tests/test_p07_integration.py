"""P07 offline smoke/negative contracts. NOT real Forge GPU acceptance."""
import base64
import json
import sys
import types
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from coherent_diffusion.forge_bridge import entrypoint as ep


class InputValidation(unittest.TestCase):
    def test_gradio_integral_number_validation(self):
        self.assertEqual(ep._int(1024.0, "Width", 256, 4096), 1024)
        self.assertEqual(ep._int("1024", "Width", 256, 4096), 1024)
        for wrong in (1024.1, float("inf"), float("nan"), True, "1e3"):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                ep._int(wrong, "Width", 256, 4096)

    def test_default_crf_roundtrip_and_hash(self):
        from coherent_diffusion.crf import canonical_json, parse_validate
        doc = ep._default_doc("Exactly one sun", "", 1024, 768)
        self.assertEqual(parse_validate(canonical_json(doc)).document, doc)
        self.assertEqual(len(doc["regions"]), 0)

    def test_binary_mask_input_failures(self):
        with self.assertRaisesRegex(ValueError, "both"):
            ep._parse_composition("{}", "aGVsbG8=", "", "", 512, 512)
        with self.assertRaisesRegex(ValueError, "encoding"):
            ep._parse_composition("", "####", "", "", 512, 512)
        with self.assertRaisesRegex(ValueError, "size"):
            ep._parse_composition("", "0" * (ep._MAX_B64_CHARS + 1), "", "", 512, 512)
        with self.assertRaises(Exception):
            ep._parse_composition("", base64.b64encode(b"not a ZIP").decode(), "", "", 512, 512)

    def test_no_loaded_host_reports_error_without_generated_image(self):
        mod=types.ModuleType("modules")
        sd_models=types.ModuleType("modules.sd_models")
        sd_models.model_data=types.SimpleNamespace(sd_model=None)
        shared=types.ModuleType("modules.shared")
        shared.state=types.SimpleNamespace(job="")
        mod.sd_models=sd_models
        mod.shared=shared
        with patch.dict(sys.modules, {"modules":mod, "modules.sd_models":sd_models,
                                      "modules.shared":shared}):
            image, status = ep.gradio_generate(
                "", "", "", "", "sdxl", "direct", 512, 512,
                512, 128, 64, 3, 1.0, 7, 512, 128)
        self.assertIsNone(image)
        self.assertEqual(status["status"], "error")
        self.assertFalse(status["partial_image"])
        self.assertIn("Load", status["detail"])


@unittest.skipIf(torch is None, "Pinned CPU torch runs these contracts in tensor job")
class RunnerContracts(unittest.TestCase):
    def plan(self, *, family="sdxl", mode="direct", scene_json="", archive_b64="", width=512,
             height=512, tile=512, overlap=128, halo=64, steps=2, cfg=1.0):
        host=types.SimpleNamespace()
        with patch("coherent_diffusion.forge_bridge.sdxl.ForgeSDXL._check_host") as sdxl, \
             patch("coherent_diffusion.forge_bridge.anima.ForgeAnima.check_host", return_value=2) as anima:
            plan=ep.preflight(host=host, family=family, mode=mode, width=width, height=height,
                              tile=tile, overlap=overlap, halo=halo, steps=steps, cfg=cfg,
                              seed=42.0, cpu_mib=512, window_mib=64, scene_json=scene_json,
                              archive_b64=archive_b64, prompt="one sun", negative="")
            if family=="sdxl":
                sdxl.assert_called_once_with(host)
            else:
                anima.assert_called_once_with(host)
        return plan

    def test_both_families_and_trajectories(self):
        for family in ("sdxl", "anima"):
            for mode in ("direct", "progressive"):
                with self.subTest(family=family, mode=mode):
                    q=self.plan(family=family,mode=mode).quality
                    self.assertEqual(q.mode,mode)
                    self.assertEqual(q.refine_sigmas[-1],0)
                    self.assertEqual(len(q.target.windows),1)
                    self.assertEqual(q.anchor is not None,mode=="progressive")

    def test_explicit_geometry_rejections(self):
        for kw in ({"width":513},{"height":4096+64},{"tile":352},
                   {"halo":72},{"overlap":512},{"steps":101}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):
                self.plan(**kw)
        with self.assertRaises(ep.UnsupportedGeneration):
            self.plan(family="anima", mode="unsupported")

    def test_crf_aspect_and_lora_safety(self):
        from coherent_diffusion.crf import canonical_json
        doc=ep._default_doc("sun","",512,512)
        doc["regions"]=[{
            "id":"a","geometry":{"type":"bbox","bbox":[0,0,1,1]},
            "prompt":"one sun","blend":{"weight":1,"feather_px":0,"z_index":1},
            "loras":[{"id":"l","name":"local","weight":1,"scope":"regional",
                      "requested_isolation":"prediction"}],
        }]
        with self.assertRaisesRegex(ep.UnsupportedGeneration,"exclusive"):
            self.plan(scene_json=canonical_json(doc))
        doc["regions"][0]["loras"]=[]
        p=self.plan(scene_json=canonical_json(doc))
        self.assertTrue(p.has_regions)
        with self.assertRaisesRegex(ValueError,"canvas"):
            self.plan(scene_json=canonical_json(doc),width=1024)

    def test_opt_in_execution_pil_metadata_and_cleanup(self):
        from coherent_diffusion.scheduler.quality import QualityResult
        plan=self.plan()
        result=types.SimpleNamespace(latent=torch.zeros(1,4,64,64),model_forwards=2)
        pixels=torch.ones((1,512,512,3),dtype=torch.float32)*0.5
        with patch("coherent_diffusion.forge_bridge.quality.run_forge_sdxl_quality",return_value=result) as run, \
             patch.object(ep,"_decode",return_value=pixels) as decode:
            image, meta=ep.execute(plan,host=types.SimpleNamespace())
        self.assertEqual(image.size,(512,512))
        self.assertEqual(meta["seed"],42)
        self.assertEqual(meta["crf_sha256"],plan.document_hash)
        self.assertEqual(meta["model_forwards_estimate"],2)
        run.assert_called_once()
        decode.assert_called_once()
        self.assertFalse(ep._JOB_LOCK.locked())

    def test_exception_and_cancellation_are_recoverable(self):
        plan=self.plan()
        with patch("coherent_diffusion.forge_bridge.quality.run_forge_sdxl_quality",
                   side_effect=MemoryError("mock OOM")):
            with self.assertRaises(MemoryError):
                ep.execute(plan,host=types.SimpleNamespace())
        self.assertFalse(ep._JOB_LOCK.locked())
        with patch("coherent_diffusion.forge_bridge.quality.run_forge_sdxl_quality") as run:
            with self.assertRaises(InterruptedError):
                ep.execute(plan,host=types.SimpleNamespace(),cancelled=lambda:True)
            run.assert_not_called()
        self.assertFalse(ep._JOB_LOCK.locked())

    def test_reentrant_job_refused(self):
        plan=self.plan()
        ep._JOB_LOCK.acquire()
        try:
            with self.assertRaisesRegex(ep.UnsupportedGeneration,"Another"):
                ep.execute(plan,host=types.SimpleNamespace())
        finally:
            ep._JOB_LOCK.release()
