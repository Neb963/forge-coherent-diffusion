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
             height=512, tile=512, overlap=128, halo=64, steps=2, cfg=1.0,
             activate_region_loras=False):
        host=types.SimpleNamespace()
        with patch("coherent_diffusion.forge_bridge.sdxl.ForgeSDXL._check_host") as sdxl, \
             patch("coherent_diffusion.forge_bridge.anima.ForgeAnima.check_host", return_value=2) as anima, \
             patch.object(ep, "_model_sigmas",
                          side_effect=lambda host, family, n: tuple(1-i/n for i in range(n+1))):
            plan=ep.preflight(host=host, family=family, mode=mode, width=width, height=height,
                              tile=tile, overlap=overlap, halo=halo, steps=steps, cfg=cfg,
                              seed=42.0, cpu_mib=512, window_mib=64, scene_json=scene_json,
                              archive_b64=archive_b64, prompt="one sun", negative="",
                              activate_region_loras=activate_region_loras)
            if family=="sdxl":
                sdxl.assert_called_once_with(host)
            else:
                anima.assert_called_once_with(host)
        return plan

    def test_native_predictor_sigma_sources_and_refusals(self):
        # The SDXL training log-sigma table is not capped at 1 by the UI.
        class Prediction:
            prediction_type="epsilon"
            sigmas=torch.linspace(.02,12.0,1000)
            def sigma(self, time):
                return .02 + time.float() / 999 * 11.98
        class PredictionFlux:
            prediction_type="const"
            def sigma(self, time):
                return (time.float()*.9).square()
        class PredictionDiscreteFlow:
            prediction_type="const"
            multiplier=1000
            def sigma(self, time):
                return time.float()/1000
        def host(p):
            return types.SimpleNamespace(forge_objects=types.SimpleNamespace(
                unet=types.SimpleNamespace(model=types.SimpleNamespace(predictor=p))))
        sd=ep._model_sigmas(host(Prediction()),"sdxl",4)
        self.assertGreater(sd[0],10)
        self.assertEqual(sd[-1],0)
        an=ep._model_sigmas(host(PredictionFlux()),"anima",4)
        self.assertAlmostEqual(an[0],.81,places=6)
        self.assertEqual(an[-1],0)
        self.assertTrue(all(a>b for a,b in zip(an,an[1:])))
        discrete=ep._model_sigmas(host(PredictionDiscreteFlow()),"anima",4)
        self.assertAlmostEqual(discrete[0],1)
        self.assertEqual(discrete[-1],0)
        with self.assertRaises(ep.UnsupportedGeneration):
            ep._model_sigmas(host(PredictionFlux()),"sdxl",4)
        with self.assertRaises(ep.UnsupportedGeneration):
            ep._model_sigmas(host(types.SimpleNamespace()),"anima",4)

    def test_both_families_and_trajectories(self):
        for family in ("sdxl", "anima"):
            for mode in ("direct", "progressive"):
                with self.subTest(family=family, mode=mode):
                    q=self.plan(family=family,mode=mode).quality
                    self.assertEqual(q.mode,mode)
                    self.assertEqual(q.refine_sigmas[-1],0)
                    self.assertEqual(len(q.target.windows),1)
                    self.assertEqual(q.anchor is not None,mode=="progressive")
                    if mode == "progressive":
                        self.assertLess(q.refine_sigmas[0], q.anchor_sigmas[0])
                        self.assertEqual(q.anchor_sigmas[0], 1.0)
                        self.assertGreater(q.refine_sigmas[0], 0)
                    else:
                        self.assertEqual(q.refine_sigmas[0], 1.0)

    def test_progressive_anima_anchor_is_not_erased_at_refine_start(self):
        from coherent_diffusion.scheduler.anima_flow import execute_anima
        q=self.plan(family="anima",mode="progressive").quality
        images=[]
        for anchor_signal in (1.0,-1.0):
            seen=[]
            def predict(crop,sigma,tile,stage):
                if stage=="anchor":
                    return torch.full_like(crop,anchor_signal)
                if not seen:
                    seen.append(crop.detach().clone())
                return crop
            result=execute_anima(q,7,predict,device="cpu",
                                 max_cpu_bytes=512*1024**2)
            self.assertTrue(seen)
            self.assertEqual(result.mode,"progressive")
            images.append(seen[0])
        self.assertGreater(float((images[0]-images[1]).abs().mean()),.3)

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
        bypassed=self.plan(scene_json=canonical_json(doc))
        self.assertEqual(bypassed.bypassed_region_loras,1)
        self.assertEqual(bypassed.document["regions"][0]["loras"],[])
        activated=self.plan(scene_json=canonical_json(doc),activate_region_loras=True)
        self.assertTrue(activated.activate_region_loras)
        self.assertEqual(len(activated.document["regions"][0]["loras"]),1)
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
        self.assertEqual(meta["window_step_evaluations_lower_bound"],2)
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
