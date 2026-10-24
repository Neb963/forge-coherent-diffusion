"""P03 real torch CPU invariants and strict source-shaped Forge SDXL fakes."""
import math
import sys
import types
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "P03 tensor suite runs in dedicated torch-cpu CI job")
class SynchronizedSDXLTests(unittest.TestCase):
    def setUp(self):
        from coherent_diffusion.scheduler import TilePlan, seeded_noise, synchronized_euler
        self.TilePlan, self.seeded_noise, self.run = TilePlan, seeded_noise, synchronized_euler
        self.plan = TilePlan.build(64, 48, 32, 16, 8)
        self.x = torch.arange(1 * 4 * 6 * 8, dtype=torch.float32).view(1, 4, 6, 8) / 100

    def test_geometry_and_halo(self):
        p = self.plan
        self.assertGreater(len(p.windows), 1)
        self.assertEqual((p.width, p.height), (8, 6))
        for t in p.windows:
            self.assertLessEqual(t.read.x0, t.write.x0)
            self.assertLessEqual(t.read.y0, t.write.y0)
            self.assertGreaterEqual(t.read.x1, t.write.x1)
            self.assertGreaterEqual(t.read.y1, t.write.y1)
            self.assertLessEqual(t.read.width - t.write.width, 2)
            self.assertLessEqual(t.read.height - t.write.height, 2)
        coverage = torch.zeros(6, 8)
        for t in p.windows:
            r = t.write
            coverage[r.y0:r.y1, r.x0:r.x1] += 1
        self.assertTrue(bool((coverage > 0).all()))
        self.assertTrue(bool((coverage > 1).any()))

    def test_geometry_invalid_and_budget(self):
        for args in [(63, 48, 32, 16, 8), (64, 48, 32, 32, 8),
                     (64, 48, 32, 16, -8), (64, 48, 32, 16, 7)]:
            with self.assertRaises(ValueError):
                self.TilePlan.build(*args)
        with self.assertRaises(ValueError):
            self.TilePlan.build(512, 512, 32, 16, 8, max_windows=3)

    def test_single_global_noise_deterministic(self):
        a = self.seeded_noise(self.plan, 42, sigma=4.2)
        b = self.seeded_noise(self.plan, 42, sigma=4.2)
        c = self.seeded_noise(self.plan, 43, sigma=4.2)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))
        self.assertEqual(tuple(a.shape), (1,4,6,8))
        self.assertEqual(a.device.type, "cpu")
        with self.assertRaises(ValueError):
            self.seeded_noise(self.plan, 1, channels=16)

    def test_pointwise_baseline_matches_untiled(self):
        calls = []
        def denoiser(crop, sigma, tile):
            calls.append((tile.index, tuple(crop.shape), sigma))
            return crop * .75 + 1
        original = self.x.clone()
        result = self.run(self.plan, self.x, [1., 0.], denoiser)
        self.assertTrue(torch.allclose(result.latent, self.x * .75 + 1, atol=1e-6))
        self.assertTrue(torch.equal(self.x, original))
        self.assertEqual(len(calls), len(self.plan.windows))
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.steps[0].windows, len(calls))
        self.assertGreater(result.cpu_workspace_estimate_bytes, 0)

    def test_identical_one_tile(self):
        p = self.TilePlan.build(64, 48, 128, 16, 8)
        self.assertEqual(len(p.windows), 1)
        result = self.run(p, self.x, [3., 0.], lambda crop, sigma, t: crop + 2)
        self.assertTrue(torch.allclose(result.latent, self.x + 2, atol=1e-6))

    def test_multi_step_global_euler_not_local_solvers(self):
        calls = []
        def predict(crop, sigma, tile):
            calls.append((sigma, tile.index))
            return crop * .5
        out = self.run(self.plan, self.x, [2., 1., 0.], predict)
        self.assertTrue(torch.allclose(out.latent, self.x * .375, atol=1e-6))
        self.assertEqual(len(calls), 2 * len(self.plan.windows))
        self.assertEqual([s.sigma for s in out.steps], [2., 1.])

    def test_order_independence_exact_for_fixed_traversal(self):
        from dataclasses import replace
        rev = replace(self.plan, windows=tuple(reversed(self.plan.windows)))
        a = self.run(self.plan, self.x, [1., 0.],
                     lambda crop, sigma, t: crop * (1 + t.index / 10)).latent
        b = self.run(rev, self.x, [1., 0.],
                     lambda crop, sigma, t: crop * (1 + t.index / 10)).latent
        self.assertTrue(torch.equal(a, b))

    def test_windows_read_frozen_master_even_if_crop_is_mutated(self):
        reads = []
        before = self.x.clone()
        def predict(crop, sigma, t):
            read = t.read
            reads.append((t, crop.clone()))
            crop.add_(1000)  # may mutate window crop, never shared global state
            return torch.ones_like(crop)
        self.run(self.plan, self.x, [1., 0.], predict)
        for tile, observed in reads:
            r = tile.read
            self.assertTrue(torch.equal(observed, before[:, :, r.y0:r.y1, r.x0:r.x1]))
        self.assertTrue(torch.equal(before, self.x))

    def test_halo_is_context_only(self):
        p = self.plan
        def denoiser(crop, sigma, tile):
            out = torch.ones_like(crop)
            q, r = tile.write, tile.read
            out[:, :, :q.y0-r.y0, :] = 100000
            out[:, :, q.y1-r.y0:, :] = 100000
            out[:, :, :, :q.x0-r.x0] = 100000
            out[:, :, :, q.x1-r.x0:] = 100000
            return out
        result = self.run(p, self.x, [1., 0.], denoiser)
        self.assertTrue(torch.allclose(result.latent, torch.ones_like(self.x), atol=1e-6))

    def test_invalid_schedule_and_preflight_cleanup(self):
        for seq in ([1., 1., 0.], [1., float("nan"), 0.], [1., .1], [0., 0.]):
            events = []
            with self.assertRaises(ValueError):
                self.run(self.plan, self.x, seq, lambda *args: None, cleanup=lambda: events.append("clean"))
            self.assertEqual(events, ["clean"])
        with self.assertRaises(MemoryError):
            self.run(self.plan, self.x, [1., 0.], lambda *args: None, max_cpu_bytes=1)
        with self.assertRaises(MemoryError):
            self.run(self.plan, self.x, [1., 0.], lambda *args: None, max_window_bytes=1)

    def test_incomplete_duplicate_plan_rejected(self):
        from dataclasses import replace
        for windows in (self.plan.windows[:-1], self.plan.windows + self.plan.windows[:1]):
            with self.assertRaises(ValueError):
                self.run(replace(self.plan, windows=windows),
                         self.x, [1., 0.], lambda *args: None)

    def test_bad_predictions_fail_atomically_and_clean(self):
        baseline = self.x.clone()
        for pred in (
            lambda crop, s, t: torch.zeros_like(crop)[:, :, :-1],
            lambda crop, s, t: torch.full_like(crop, float("nan")),
            lambda crop, s, t: 17,
        ):
            event = []
            with self.assertRaises(ValueError):
                self.run(self.plan, self.x, [1.,0.], pred,
                         cleanup=lambda: event.append("done"))
            self.assertEqual(event, ["done"])
            self.assertTrue(torch.equal(self.x, baseline))

    def test_interrupt_mid_step_and_progress_fail_clean(self):
        baseline = self.x.clone()
        count = [0]
        def stop():
            count[0] += 1
            return count[0] >= 3
        finished = []
        with self.assertRaises(InterruptedError):
            self.run(self.plan, self.x, [1.,0.], lambda c,s,t: c,
                     cancelled=stop, cleanup=lambda: finished.append(True))
        self.assertEqual(finished, [True])
        with self.assertRaises(RuntimeError):
            self.run(self.plan, self.x, [1.,0.], lambda c,s,t:c,
                     progress=lambda x: (_ for _ in ()).throw(RuntimeError("callback failure")),
                     cleanup=lambda: finished.append(True))
        self.assertEqual(finished, [True, True])
        self.assertTrue(torch.equal(self.x, baseline))

    def test_device_transfer_and_shape(self):
        seen=[]
        result=self.run(self.plan,self.x,[1.,0.],
                        lambda crop,sigma,t: (seen.append((str(crop.device),crop.dtype)),crop)[1],
                        device="cpu",inference_dtype=torch.bfloat16)
        self.assertTrue(torch.allclose(result.latent, self.x, atol=1e-6))
        self.assertEqual({v[1] for v in seen},{torch.bfloat16})


@unittest.skipIf(torch is None, "P03 adapter contract runs in dedicated torch-cpu CI job")
class ForgeSDXLBridgeTests(unittest.TestCase):
    def setUp(self):
        from coherent_diffusion.forge_bridge.sdxl import ForgeSDXL, UnsupportedSDXL
        from coherent_diffusion.scheduler import TilePlan
        self.ForgeSDXL, self.UnsupportedSDXL = ForgeSDXL, UnsupportedSDXL
        self.plan = TilePlan.build(32, 32, 32, 0, 8)
        self.calls = []
        calls = self.calls
        cond = types.ModuleType("backend.sampling.condition")
        cond.compile_conditions = lambda value: [{"model_conds": {"c_crossattn": value["crossattn"],
                                                                  "y": value["vector"]}}]
        sampling = types.ModuleType("backend.sampling.sampling_function")
        sampling.sampling_prepare = lambda unet, x: calls.append(("prepare",tuple(x.shape),str(x.device)))
        sampling.sampling_cleanup = lambda unet: calls.append(("cleanup",))
        def fake_inner(model, crop, t, uncond, pos, cfg, opts, seed):
            calls.append(("predict", tuple(crop.shape), float(t[0]), cfg, seed))
            return crop * .25 + 1
        sampling.sampling_function_inner = fake_inner
        self.patches = patch.dict(sys.modules, {
            "backend": types.ModuleType("backend"),
            "backend.sampling": types.ModuleType("backend.sampling"),
            "backend.sampling.condition": cond,
            "backend.sampling.sampling_function": sampling,
        })
        self.patches.start()
        self.addCleanup(self.patches.stop)

        class StableDiffusionXL:
            is_sdxl = True
            is_inpaint = False
            use_shift = False
            def __init__(self):
                net = types.SimpleNamespace(
                    apply_model=lambda *a, **kw: None)
                unet = types.SimpleNamespace(model=net,model_options={},
                                             controlnet_linked_list=None,extra_concat_condition=None)
                vaemodel=types.SimpleNamespace(process_out=lambda t: t + 9)
                vae=types.SimpleNamespace(latent_channels=4,upscale_ratio=8,
                                          first_stage_model=vaemodel,
                                          decode_tiled=lambda t, **kw: (calls.append(("decode", kw)),
                                                                      t.movedim(1,-1))[1])
                self.forge_objects=types.SimpleNamespace(unet=unet,vae=vae)
            def get_learned_conditioning(self, prompts):
                calls.append(("encode",prompts[0],prompts.width,prompts.height,prompts.is_negative_prompt))
                return {"crossattn":torch.ones((1,2,4)), "vector":torch.zeros((1,8))}
        self.host=StableDiffusionXL()

    def test_source_shaped_entrypoints_and_tiled_decode(self):
        a=self.ForgeSDXL.from_prompts(self.host,self.plan,prompt="sun",
                                      negative_prompt="blur",cfg=4.5,seed=91,device="cpu")
        self.assertEqual([t[:2] for t in self.calls[:2]],[("encode","sun"),("encode","blur")])
        result=a.run([1., 0.])
        self.assertEqual(len(result.steps),1)
        self.assertEqual([c[0] for c in self.calls if c[0] in ("prepare","predict","cleanup")],
                         ["prepare","predict","cleanup"])
        self.assertEqual([c[2] for c in self.calls if c[0]=="prepare"],["meta"])
        out=a.decode_tiled(result.latent, tile_latent=16, overlap_latent=4)
        self.assertEqual(tuple(out.shape),(1,4,4,4))
        self.assertTrue(torch.all(out > 8))
        self.assertEqual(self.calls[-1], ("decode", {"tile_x":16,"tile_y":16,"overlap":4}))

    def test_host_denies_unsupported_configurations(self):
        cases=[
            ("is_inpaint",True),("use_shift",True),
        ]
        for name,value in cases:
            setattr(self.host,name,value)
            with self.assertRaises(self.UnsupportedSDXL):
                self.ForgeSDXL.from_prompts(self.host,self.plan,prompt="",negative_prompt="",cfg=1.,seed=1)
            setattr(self.host,name,False)
        unet=self.host.forge_objects.unet
        unet.model_options={"sampler_cfg_function": object()}
        with self.assertRaises(self.UnsupportedSDXL):
            self.ForgeSDXL.from_prompts(self.host,self.plan,prompt="",negative_prompt="",cfg=1,seed=1)
        unet.model_options={}
        unet.controlnet_linked_list=object()
        with self.assertRaises(self.UnsupportedSDXL):
            self.ForgeSDXL.from_prompts(self.host,self.plan,prompt="",negative_prompt="",cfg=1,seed=1)

    def test_adapter_cleans_on_denoiser_failure(self):
        bridge=self.ForgeSDXL.from_prompts(self.host,self.plan,prompt="x",negative_prompt="",cfg=2,seed=1,device="cpu")
        module=sys.modules["backend.sampling.sampling_function"]
        module.sampling_function_inner=lambda *args: (_ for _ in ()).throw(RuntimeError("forced fake OOM"))
        with self.assertRaisesRegex(RuntimeError,"forced fake OOM"):
            bridge.run([1.,0.])
        self.assertFalse(bridge._prepared)
        self.assertEqual([c[0] for c in self.calls if c[0] in ("prepare","cleanup")],["prepare","cleanup"])


if __name__ == "__main__":
    unittest.main()
