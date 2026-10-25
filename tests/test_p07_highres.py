"""P07 high-resolution stress contracts: planning only; never allocate GPU canvas.

Verify no fixed output/anchor pixel ceiling. A model forward always sees at
most the selected read-window's patch-token count; CPU budgets remain enforced
at execution. Real image/VRAM evidence is still mandatory in P07.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "pinned tensor job validates high-res geometry")
class HighResContracts(unittest.TestCase):
    def test_both_families_multiple_large_aspect_ratios(self):
        from coherent_diffusion.scheduler.sdxl_sync import TilePlan
        from coherent_diffusion.scheduler.quality import QualityPlan
        from coherent_diffusion.scheduler.anima_flow import AnimaQualityPlan, validate_anima_tiles
        from coherent_diffusion.scheduler.anchor import expected_anchor_dimensions
        for w, h in [(4096,4096), (4096,3072), (8192,6144),
                     (16384,16384), (16384,256)]:
            with self.subTest(canvas=(w,h)):
                target = TilePlan.build(w, h, 512, 128, 64)
                area = max(t.read.width*t.read.height for t in target.windows)
                area *= target.scale**2
                for family in ("sdxl","anima"):
                    if family == "sdxl":
                        plan = QualityPlan.build(target, mode="progressive",
                                                 refine_sigmas=(.5,0),
                                                 anchor_sigmas=(1.,0.))
                        factor, alignment = plan.anchor_factor, 8
                    else:
                        plan = AnimaQualityPlan.build(target, mode="progressive",
                                                      patch_spatial=2, factor=2,
                                                      refine_sigmas=(.5,0),
                                                      anchor_sigmas=(1.,0.))
                        factor, alignment = plan.factor, 16
                        validate_anima_tiles(plan.anchor, patch_spatial=2)
                    self.assertGreaterEqual(factor, 2)
                    self.assertLessEqual(plan.anchor.width_px*plan.anchor.height_px,area)
                    self.assertEqual((plan.anchor.width_px,plan.anchor.height_px),
                                     expected_anchor_dimensions(target,factor,alignment))
                    self.assertEqual(len(plan.anchor.windows),1)
                    self.assertEqual(plan.target.width_px,w)
                    self.assertEqual(plan.target.height_px,h)

    def test_anima_anchor_one_step_fails_only_real_cpu_budget(self):
        from coherent_diffusion.scheduler.sdxl_sync import TilePlan
        from coherent_diffusion.scheduler.anima_flow import AnimaQualityPlan,execute_anima
        target=TilePlan.build(4096,4096,512,128,64)
        plan=AnimaQualityPlan.build(target,patch_spatial=2,mode="progressive",
                                    refine_sigmas=(.5,0),anchor_sigmas=(1,0))
        with self.assertRaisesRegex(MemoryError,"CPU"):
            execute_anima(plan,seed=1,predict=lambda *args:None,
                          max_cpu_bytes=4096,max_window_bytes=512*1024**2)

    def test_explicit_tiny_global_model_window_fails_instead_of_erasing_context(self):
        from coherent_diffusion.scheduler.sdxl_sync import TilePlan
        from coherent_diffusion.scheduler.anima_flow import AnimaQualityPlan
        t=TilePlan.build(4096,4096,512,128,64)
        with self.assertRaisesRegex(MemoryError,"No patch-aligned global overview"):
            AnimaQualityPlan.build(t,patch_spatial=2,mode="progressive",
                                   refine_sigmas=(.5,0),anchor_sigmas=(1,0),
                                   max_anchor_pixels=1)

    def test_optional_tile_count_limit_and_region_workspace(self):
        from coherent_diffusion.scheduler.sdxl_sync import TilePlan
        t=TilePlan.build(16384,16384,256,192,64)
        self.assertGreater(len(t.windows),4096)
        with self.assertRaisesRegex(ValueError,"Tile count"):
            TilePlan.build(16384,16384,256,192,64,max_windows=4096)
        from coherent_diffusion.scheduler.regional import compile_regional
        from coherent_diffusion.forge_bridge.entrypoint import _default_doc
        doc=_default_doc("one sun","",16384,16384)
        # No artificial 1,048,576 latent-pixel refusal for a base-only scene.
        # Do not allocate the large coefficient tensors here; verify the
        # compiler's signature and preflight a smaller real grid.
        import inspect
        self.assertIsNone(inspect.signature(compile_regional)
                          .parameters["max_latent_pixels"].default)
        t_small=TilePlan.build(512,512,512,128,64)
        states=compile_regional(_default_doc("sun","",512,512),t_small)
        self.assertEqual(len(states.states),1)
        with self.assertRaises(MemoryError):
            compile_regional(_default_doc("sun","",512,512),t_small,
                             max_mask_workspace_bytes=1)

    def test_regional_forward_count_trades_time_not_resolution(self):
        from coherent_diffusion.scheduler.sdxl_sync import TilePlan
        from coherent_diffusion.scheduler.regional import compile_regional, UnsupportedRegional
        from coherent_diffusion.forge_bridge.entrypoint import _default_doc
        doc = _default_doc("city skyline", "", 256, 256)
        for i in range(20):
            doc["regions"].append({
                "id": f"r{i}", "geometry": {"type": "bbox", "bbox": [0,0,1,1]},
                "prompt": f"distinct material {i}",
                "blend": {"weight":1, "feather_px":0, "z_index":i},
                "loras": []
            })
        target = TilePlan.build(256,256,256,64,0)
        compiled = compile_regional(doc, target)
        self.assertEqual(len(compiled.states),21)
        self.assertEqual(compiled.max_forwards_per_window,21)
        with self.assertRaisesRegex(UnsupportedRegional, "explicit forward allowance"):
            compile_regional(doc,target,max_forwards_per_window=16)

    def test_p07_preflight_canvas_limit_matches_crf_and_editor(self):
        from coherent_diffusion.forge_bridge import entrypoint as ep
        from coherent_diffusion.crf import parse_validate,canonical_json
        from coherent_diffusion.scheduler.anima_flow import AnimaQualityPlan
        from coherent_diffusion.forge_bridge.anima import ForgeAnima
        import types
        fake=types.SimpleNamespace()
        doc=ep._default_doc("coherent cityscape","",8192,6144)
        parse_validate(canonical_json(doc))
        with patch.object(ForgeAnima,"check_host",return_value=2), \
             patch.object(ep,"_model_sigmas",side_effect=lambda host,family,n:(
                  1.0,0.5,0.0)):
            plan=ep.preflight(host=fake,family="anima",mode="progressive",
                              width=8192,height=6144,tile=512,overlap=128,halo=64,
                              steps=2,cfg=1.0,seed=1,cpu_mib=4096,window_mib=512,
                              scene_json=canonical_json(doc))
            self.assertEqual(plan.width,8192)
            self.assertTrue(plan.quality.anchor.width_px<plan.width)

if __name__=="__main__":
    unittest.main()
