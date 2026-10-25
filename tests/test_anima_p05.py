"""P05 deterministic CPU-only Anima flow, RoPE, Q4-preserving Forge contracts."""
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "P05 uses the pinned CPU torch CI job")
class AnimaFlowTests(unittest.TestCase):
    def setUp(self):
        from coherent_diffusion.scheduler.sdxl_sync import TilePlan
        self.plan = TilePlan.build(128, 96, 64, 16, 16)

    def test_flow_5d_frozen_global_state_and_one_global_step(self):
        from coherent_diffusion.scheduler.anima_flow import (
            anima_noise, synchronized_flow,
        )
        a = anima_noise(self.plan, 4321, 1.0)
        self.assertEqual(a.shape, (1, 16, 1, 12, 16))
        first = a.clone()
        observed = []
        def predict(crop, sigma, tile):
            self.assertEqual(crop.shape[1:3], (16, 1))
            observed.append((sigma, tile.index))
            return crop * 0.5
        result = synchronized_flow(self.plan, a, (1, .5, 0), predict,
                                   patch_spatial=2)
        self.assertEqual(len(observed), len(self.plan.windows) * 2)
        # For x0=x/2, each global Euler interval performs its own update.
        self.assertTrue(torch.allclose(result.latent, first * 0.375, atol=1e-6))
        self.assertTrue(torch.equal(a, first))
        self.assertEqual(len(result.steps), 2)
        self.assertEqual(result.steps[-1].next_sigma, 0)

    def test_constant_flow_velocity_and_exact_sigma_timeline(self):
        from coherent_diffusion.scheduler.anima_flow import anima_noise, synchronized_flow
        sigmas = (1.0, 0.75, 0.3, 0.0)
        v = anima_noise(self.plan, 111, 1.0)
        sequence = []
        def native_kmodel_x0(crop, sigma, tile):
            sequence.append((sigma,tile.index))
            # Forge KModel prediction_type="const": x0=x-sigma*velocity.
            return crop - sigma * 0.125
        r = synchronized_flow(self.plan, v, sigmas, native_kmodel_x0,
                              patch_spatial=2)
        self.assertEqual([x.sigma for x in r.steps], list(sigmas[:-1]))
        for step, sigma in enumerate(sigmas[:-1]):
            self.assertEqual(
                sequence[step*len(self.plan.windows):(step+1)*len(self.plan.windows)],
                [(sigma,t.index) for t in self.plan.windows])
        self.assertTrue(torch.allclose(r.latent, v - 0.125, atol=1e-6))

    def test_order_independent_and_pointwise_untiled_baseline(self):
        from coherent_diffusion.scheduler.anima_flow import (
            anima_noise, synchronized_flow,
        )
        from dataclasses import replace
        p, v = self.plan, anima_noise(self.plan, 42, 1)
        b = replace(p, windows=tuple(reversed(p.windows)))
        pred = lambda crop, sigma, t: crop * 0.2 + 0.01
        a = synchronized_flow(p, v, [1, 0], pred, patch_spatial=2)
        c = synchronized_flow(b, v, [1, 0], pred, patch_spatial=2)
        self.assertTrue(torch.equal(a.latent, c.latent))
        self.assertTrue(torch.allclose(a.latent, pred(v, 1, None), atol=1e-6))

    def test_bad_schedule_shape_patch_and_memory_rejected(self):
        from coherent_diffusion.scheduler.anima_flow import (
            flow_sigmas, synchronized_flow, anima_noise,
        )
        from coherent_diffusion.scheduler.sdxl_sync import TilePlan
        for schedule in ([2,0], [1,1,0], [1,-0.1,0], [1,float("nan"),0],
                         [1,.2], [0], [1, float("inf"),0]):
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                flow_sigmas(schedule)
        v = anima_noise(self.plan, 3, 1)
        with self.assertRaises(ValueError):
            synchronized_flow(self.plan, v, [1,0], lambda c,s,t:c,
                              patch_spatial=3)
        with self.assertRaises(MemoryError):
            synchronized_flow(self.plan, v, [1,0], lambda c,s,t:c,
                              patch_spatial=2, max_window_bytes=10)
        with self.assertRaises(ValueError):
            synchronized_flow(self.plan, v.squeeze(2), [1,0], lambda c,s,t:c,
                              patch_spatial=2)
        odd = TilePlan.build(120, 96, 64, 16, 8)
        with self.assertRaises(ValueError):
            synchronized_flow(odd, anima_noise(odd, 3, 1), [1,0],
                              lambda c,s,t:c, patch_spatial=2)

    def test_interrupt_predict_failure_and_nonfinite_cleanup(self):
        from coherent_diffusion.scheduler.anima_flow import anima_noise, synchronized_flow
        a = anima_noise(self.plan, 3, 1)
        for what in ("cancel", "fail", "nan"):
            with self.subTest(what=what):
                cleanup = []
                def pred(c,s,t):
                    if what == "fail":
                        raise RuntimeError("synthetic fail")
                    if what == "nan":
                        return torch.full_like(c, float("nan"))
                    return c
                with self.assertRaises((InterruptedError, RuntimeError, ValueError)):
                    synchronized_flow(self.plan, a, [1,0], pred, patch_spatial=2,
                                      cancelled=(lambda:True) if what == "cancel" else None,
                                      cleanup=lambda:cleanup.append("closed"))
                self.assertEqual(cleanup, ["closed"])
                self.assertTrue(torch.equal(a, anima_noise(self.plan, 3, 1)))

    def test_quality_direct_progressive_reproducibility_and_handoff(self):
        from coherent_diffusion.scheduler.anima_flow import (
            AnimaQualityPlan, execute_anima,
        )
        direct = AnimaQualityPlan.build(self.plan, patch_spatial=2,
                                        mode="direct", refine_sigmas=[1,.5,0])
        prog = AnimaQualityPlan.build(self.plan, patch_spatial=2,
                                      mode="progressive", factor=2,
                                      anchor_sigmas=[1,0], refine_sigmas=[.3,0])
        stages = []
        def pred(c,s,t,stage):
            stages.append((stage, t.index))
            return c * 0.4
        a = execute_anima(direct, 54, pred)
        b = execute_anima(direct, 54, pred)
        self.assertTrue(torch.equal(a.latent,b.latent))
        self.assertEqual([s[0] for s in a.stages],["direct"])
        p = execute_anima(prog,54,pred)
        self.assertEqual([s[0] for s in p.stages], ["anchor","refine"])
        self.assertEqual(p.stages[0][1].plan.windows[0].write.width,8)
        self.assertEqual(p.latent.shape,(1,16,1,12,16))
        self.assertTrue(torch.equal(p.latent,execute_anima(prog,54,pred).latent))
        self.assertFalse(torch.equal(p.latent,a.latent))
        with self.assertRaises(MemoryError):
            execute_anima(prog,54,pred,max_cpu_bytes=100)
        with self.assertRaises(ValueError):
            AnimaQualityPlan.build(self.plan,patch_spatial=2,mode="progressive",
                                   anchor_sigmas=[1,0],refine_sigmas=[1,0],
                                   global_interaction="fft")

    def test_progressive_handoff_uses_native_flow_interpolation(self):
        from coherent_diffusion.scheduler.anima_flow import (
            AnimaQualityPlan, anima_noise, execute_anima,
        )
        sigma = 0.4
        seed = 123
        q = AnimaQualityPlan.build(
            self.plan, patch_spatial=2, mode="progressive",
            anchor_sigmas=[1, 0], refine_sigmas=[sigma, 0])
        # A constant x0 predictor makes the anchor exactly 0.1 everywhere.
        # Native constant-flow noise_scaling: (1-s)*latent + s*noise.
        target_noise = anima_noise(
            self.plan, seed ^ 0x1E3779B97F4A7C15, sigma)
        expected = 0.1 * (1.0 - sigma) + target_noise
        inspected = []
        def pred(crop, s, tile, stage):
            if stage == "refine":
                r = tile.read
                wanted = expected[..., r.y0:r.y1, r.x0:r.x1]
                self.assertTrue(torch.allclose(crop, wanted, atol=1e-6))
                inspected.append(tile.index)
            return torch.full_like(crop, 0.1)
        execute_anima(q, seed, pred)
        self.assertEqual(len(inspected), len(self.plan.windows))

    def test_quality_partial_stage_failure_releases_once(self):
        from coherent_diffusion.scheduler.anima_flow import AnimaQualityPlan,execute_anima
        q=AnimaQualityPlan.build(self.plan,patch_spatial=2,mode="progressive",
                                 anchor_sigmas=[1,0],refine_sigmas=[.5,0])
        closed=[]
        def fail(c,s,t,stage):
            if stage=="refine":
                raise RuntimeError("refinement")
            return c
        with self.assertRaisesRegex(RuntimeError,"refinement"):
            execute_anima(q,1,fail,cleanup=lambda:closed.append(1))
        self.assertEqual(closed,[1])


@unittest.skipIf(torch is None, "P05 uses pinned torch")
class GlobalRopeTests(unittest.TestCase):
    class FakeEmbedder:
        def __init__(self):
            self.dim_spatial_range = torch.arange(0, 8, 2).float()/8
            self.dim_temporal_range = torch.arange(0, 8, 2).float()/8
            self.h_ntk_factor = 1.25
            self.w_ntk_factor = 1.5
            self.t_ntk_factor = 1.0

    def test_global_crop_matches_absolute_full_token_grid(self):
        from coherent_diffusion.forge_bridge.anima_position import absolute_video_rope
        e = self.FakeEmbedder()
        full = absolute_video_rope(e,t_tokens=1,h_tokens=12,w_tokens=16,
                                   y_patch=0,x_patch=0,device="cpu")
        block = absolute_video_rope(e,t_tokens=1,h_tokens=4,w_tokens=5,
                                    y_patch=3,x_patch=7,device="cpu")
        f = full.reshape(1,12,16,-1,2,2)[:,3:7,7:12].reshape(block.shape)
        self.assertTrue(torch.allclose(block, f, atol=0,rtol=0))
        self.assertEqual(tuple(block.shape),(20,12,2,2))

    def test_scoped_position_restore_even_if_forward_raises(self):
        from coherent_diffusion.forge_bridge.anima_position import scoped_absolute_rope
        from coherent_diffusion.scheduler.sdxl_sync import TilePlan
        tile = TilePlan.build(128,96,64,16,16).windows[-1]
        Emb = type("VideoRopePosition3DEmb",(self.FakeEmbedder,),{})
        def orig(self,x,padding_mask=None):
            return torch.zeros((1,1,x.shape[-2]//2,x.shape[-1]//2,48)), (
                torch.zeros((x.shape[-2]//2)*(x.shape[-1]//2),12,2,2)),None
        Core = type("Anima",(),{"__module__":"backend.nn.anima",
                                "prepare_embedded_sequence":orig})
        core=Core()
        core.patch_spatial=2
        core.patch_temporal=1
        core.pos_embedder=Emb()
        with self.assertRaisesRegex(RuntimeError,"boom"):
            with scoped_absolute_rope(core,tile,patch_spatial=2):
                result=core.prepare_embedded_sequence(
                    torch.zeros((1,16,1,tile.read.height,tile.read.width)))
                self.assertNotEqual(float(result[1].abs().sum()),0)
                with self.assertRaisesRegex(Exception,"Concurrent"):
                    with scoped_absolute_rope(core,tile,patch_spatial=2):
                        pass
                raise RuntimeError("boom")
        self.assertNotIn("prepare_embedded_sequence",vars(core))
        self.assertNotIn("_coherent_rope_active",vars(core))
        self.assertEqual(float(core.prepare_embedded_sequence(
            torch.zeros((1,16,1,4,4)))[1].abs().sum()),0)


@unittest.skipIf(torch is None, "P05 uses pinned torch")
class ForgeAnimaTests(unittest.TestCase):
    def setUp(self):
        from coherent_diffusion.scheduler.sdxl_sync import TilePlan
        self.plan=TilePlan.build(128,96,64,16,16)
        self.events=[]
        events=self.events
        FakeEmbedder=type("VideoRopePosition3DEmb",(),{
            "__init__":lambda self:None,
            "dim_spatial_range":torch.arange(0,8,2).float()/8,
            "dim_temporal_range":torch.arange(0,8,2).float()/8,
            "h_ntk_factor":1.0,"w_ntk_factor":1.0,"t_ntk_factor":1.0,
        })
        def prepared(self,x,padding_mask=None):
            h,w=x.shape[-2]//2,x.shape[-1]//2
            return torch.zeros((x.shape[0],1,h,w,48)),torch.zeros((h*w,12,2,2)),None
        Core=type("Anima",(),{
            "__module__":"backend.nn.anima",
            "prepare_embedded_sequence":prepared,
            "patch_spatial":2,"patch_temporal":1,"in_channels":16,"out_channels":16,
        })
        core=Core()
        core.pos_embedder=FakeEmbedder()
        predictor=type("PredictionDiscreteFlow",(),{"prediction_type":"const"})()
        model=SimpleNamespace(diffusion_model=core,predictor=predictor,
                              computation_dtype=torch.float32,apply_model=lambda *x:None)
        vae=SimpleNamespace(is_wan=True,latent_dim=3,latent_channels=16,
                            upscale_ratio=(lambda n:n*4-3,8,8),
                            first_stage_model=SimpleNamespace(process_out=lambda x:x*2),
                            decode_tiled=lambda x,**kw: self.decode(x,kw))
        clip=SimpleNamespace(gguf_parameter_marker=True)
        unet=SimpleNamespace(model=model,controlnet_linked_list=None,
                             extra_concat_condition=None,model_options={})
        def enc(prompt):
            events.append(("encode",prompt[0],prompt.is_negative_prompt))
            return [torch.ones((1,512,8))]
        Host=type("Anima",(),{"is_wan":True,"is_sdxl":False,"is_inpaint":False,
                              "ini_latent":None,"ref_latents":[],
                              "get_learned_conditioning":lambda _,p:enc(p)})
        self.host=Host()
        self.host.forge_objects=SimpleNamespace(unet=unet,vae=vae,clip=clip)
        self.original_clip=clip
        sampling=types.ModuleType("backend.sampling.sampling_function")
        sampling.sampling_prepare=lambda u,x:events.append(("prepare",tuple(x.shape)))
        sampling.sampling_cleanup=lambda u:events.append(("cleanup",))
        def sample(m,crop,s,neg,pos,cfg,options,seed):
            events.append(("sample",float(s[0]),tuple(crop.shape)))
            rope=m.diffusion_model.prepare_embedded_sequence(crop)[1]
            self.assertNotEqual(float(rope.abs().sum()),0)
            return crop*.5
        sampling.sampling_function_inner=sample
        condition=types.ModuleType("backend.sampling.condition")
        condition.compile_conditions=lambda x:[{"cross":x}]
        self.mod_patch=patch.dict(sys.modules,{
            "backend":types.ModuleType("backend"),
            "backend.sampling":types.ModuleType("backend.sampling"),
            "backend.sampling.sampling_function":sampling,
            "backend.sampling.condition":condition,
        })
        self.mod_patch.start()
        self.addCleanup(self.mod_patch.stop)

    def decode(self,x,kw):
        self.events.append(("decode",tuple(x.shape),kw))
        return torch.zeros((x.shape[0],1,x.shape[-2]*8,x.shape[-1]*8,3))

    def test_encoder_identity_shape_flow_and_native_wan_decode(self):
        from coherent_diffusion.forge_bridge.anima import ForgeAnima
        from coherent_diffusion.scheduler.anima_flow import anima_noise
        a=ForgeAnima.from_prompts(self.host,self.plan,prompt="sun",
                                  negative_prompt="duplicates",cfg=1,seed=8,device="cpu")
        self.assertIs(self.host.forge_objects.clip,self.original_clip)
        self.assertEqual([e[2] for e in self.events if e[0]=="encode"],[False,True])
        self.assertEqual(a.positive[0]["cross"].shape,(1,512,8))
        a.prepare()
        tile=self.plan.windows[-1]
        crop=anima_noise(self.plan,8,1)[...,tile.read.y0:tile.read.y1,tile.read.x0:tile.read.x1]
        out=a.predict(crop,.8,tile)
        self.assertTrue(torch.equal(out,crop*.5))
        self.assertNotIn("prepare_embedded_sequence",vars(
            self.host.forge_objects.unet.model.diffusion_model))
        a.cleanup()
        self.assertEqual(len([x for x in self.events if x[0]=="cleanup"]),1)
        pixels=a.decode_tiled(anima_noise(self.plan,8,1))
        self.assertEqual(tuple(pixels.shape),(1,96,128,3))
        self.assertEqual([e[2] for e in self.events if e[0]=="decode"][0],
                         {"tile_x":32,"tile_y":32,"overlap":8})

    def test_quality_stages_cache_encoder_and_cleanup(self):
        from coherent_diffusion.scheduler.anima_flow import AnimaQualityPlan
        from coherent_diffusion.forge_bridge.anima import run_forge_anima_quality
        q=AnimaQualityPlan.build(self.plan,patch_spatial=2,mode="progressive",
                                 anchor_sigmas=[1,0],refine_sigmas=[.25,0])
        r=run_forge_anima_quality(self.host,q,prompt="one lighthouse",
                                  seed=37,device="cpu")
        self.assertEqual(r.mode,"progressive")
        self.assertEqual(len([e for e in self.events if e[0]=="encode"]),2)
        self.assertEqual(len([e for e in self.events if e[0]=="prepare"]),2)
        self.assertEqual(len([e for e in self.events if e[0]=="cleanup"]),2)
        self.assertEqual(len([e for e in self.events if e[0]=="sample"]),r.model_forwards)

    def test_unsupported_configurations_refuse_no_mutation(self):
        from coherent_diffusion.forge_bridge.anima import ForgeAnima, UnsupportedAnima
        cases=[
            ("refs", lambda: setattr(self.host,"ref_latents",[torch.zeros(1)])),
            ("hook", lambda: setattr(self.host.forge_objects.unet,"model_options",{"patches":{}})),
            ("vae",lambda:setattr(self.host.forge_objects.vae,"latent_channels",4)),
            ("predictor",lambda:setattr(self.host.forge_objects.unet.model.predictor,
                                        "prediction_type","epsilon")),
        ]
        for name,modify in cases:
            with self.subTest(name=name):
                from copy import deepcopy
                old_refs=self.host.ref_latents
                old_options=self.host.forge_objects.unet.model_options
                old_channels=self.host.forge_objects.vae.latent_channels
                old_pred=self.host.forge_objects.unet.model.predictor.prediction_type
                modify()
                with self.assertRaises(UnsupportedAnima):
                    ForgeAnima.from_prompts(self.host,self.plan,prompt="x",seed=3,device="cpu")
                self.host.ref_latents=old_refs
                self.host.forge_objects.unet.model_options=old_options
                self.host.forge_objects.vae.latent_channels=old_channels
                self.host.forge_objects.unet.model.predictor.prediction_type=old_pred
        self.assertEqual(self.events,[])

    def test_preparation_failure_runs_native_cleanup(self):
        from coherent_diffusion.forge_bridge.anima import ForgeAnima
        a=ForgeAnima.from_prompts(self.host,self.plan,prompt="x",seed=1,device="cpu")
        import sys
        module=sys.modules["backend.sampling.sampling_function"]
        with patch.object(module,"sampling_prepare",side_effect=RuntimeError("synthetic OOM")):
            with self.assertRaisesRegex(RuntimeError,"synthetic OOM"):
                a.prepare()
        self.assertFalse(a._prepared)
        self.assertEqual(len([e for e in self.events if e[0]=="cleanup"]),1)


if __name__ == "__main__":
    unittest.main()
