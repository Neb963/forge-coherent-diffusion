"""P07 targeted count-one entity guidance at the global anchor.

No real model is loaded. Assertions prove spatial x0 routing and explicit
native LoRA exclusion, not final Anima image adherence or prompt semantics.
"""
import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch required for regional anchor tests")
class AnchorEntityGuidance(unittest.TestCase):
    def setUp(self):
        from coherent_diffusion.scheduler.sdxl_sync import TilePlan
        self.geometry = TilePlan.build(512, 384, 512, 0, 0, scale=8)
        self.doc = {
            "format": "coherent-regions", "version": 1,
            "canvas": {"width": 2048, "height": 1536,
                       "coordinate_system": "normalized_top_left",
                       "aspect_policy": "preserve_relative"},
            "global": {"scene_prompt": "Wide coastline with one sun",
                       "style_prompt": "Natural landscape photo",
                       "negative_prompt": "multiple suns", "base_weight": 0.25},
            "blend": {"policy": "weighted"},
            "regions": [
                {"id": "sky", "geometry": {"type": "bbox", "bbox": [0, 0, 1, .46]},
                 "blend": {"weight": 1, "feather_px": 48, "z_index": 10},
                 "name": "Sky", "prompt": "thin sky", "negative_prompt": "",
                 "inherit_style": True, "enabled": True, "visible": True, "loras": []},
                {"id": "sun", "geometry": {"type": "bbox",
                                           "bbox": [.71, .08, .13, .16]},
                 "blend": {"weight": 2, "feather_px": 28, "z_index": 20},
                 "name": "Sun", "prompt": "one small bright sun",
                 "negative_prompt": "multiple suns", "inherit_style": True,
                 "enabled": True, "visible": True,
                 "loras": [{"id": "decorative", "name": "example_local_texture_lora",
                            "weight": .6, "scope": "regional",
                            "requested_isolation": "approximate"}]},
            ],
            "entities": [{"id": "sun-1", "description": "single sun", "count": 1,
                          "region_ids": ["sun"]}]
        }

    def test_anchor_compiler_concentrates_unique_entity_and_preserves_crf(self):
        from coherent_diffusion.scheduler.regional import (
            compile_entity_anchor, anchor_entity_region_ids, blended_prediction,
            UnsupportedRegional)
        before = json.dumps(self.doc, sort_keys=True)
        self.assertEqual(anchor_entity_region_ids(self.doc), ("sun",))
        out = compile_entity_anchor(self.doc, self.geometry)
        self.assertEqual(json.dumps(self.doc, sort_keys=True), before)
        self.assertEqual(len(out.states), 2, "base and one entity, no sky prompt")
        self.assertTrue(all(not s.state.adapters for s in out.states),
                        "CRF LoRAs must not patch the anchor")
        self.assertEqual((out.geometry.width_px, out.geometry.height_px), (512,384))
        region = next(s for s in out.states if "one small bright sun" in s.state.prompt)
        baseline = next(s for s in out.states if "Wide coastline" in s.state.prompt)
        self.assertIn("Natural landscape", region.state.prompt)
        self.assertAlmostEqual(float(region.weight[8, 3]), 0.0, places=5)
        # Requested sun bbox x=.71-.84, y=.08-.24; anchor pixel center
        # x=.775*512 and y=.16*384 -> latent x=49.5 y=7.5.
        self.assertGreater(float(region.weight[8, 49]), .70)
        self.assertLess(float(baseline.weight[8, 49]), .30)
        crop = torch.zeros((1,16,1,self.geometry.height,self.geometry.width))
        tile = self.geometry.windows[0]
        def pred(state, crop, sigma, tile, stage):
            value = 0.9 if "one small bright sun" in state.prompt else 0.1
            return torch.full_like(crop, value)
        with self.assertRaises(UnsupportedRegional):
            blended_prediction(out, tile, crop, .5, "anchor", pred)
        blended = blended_prediction(out, tile, crop, .5, "anchor", pred,
                                     allow_anchor=True)
        self.assertAlmostEqual(float(blended[0,0,0,8,3]), .1, places=5)
        self.assertGreater(float(blended[0,0,0,8,49]), .65)

    def test_nonunique_missing_or_inactive_entity_does_not_change_anchor(self):
        from coherent_diffusion.scheduler.regional import (
            compile_entity_anchor, anchor_entity_region_ids)
        for mutate in (
            lambda d: d.update(entities=[]),
            lambda d: d["entities"][0].update(count=2),
            lambda d: d["regions"][1].update(enabled=False),
            lambda d: d["regions"][1]["blend"].update(weight=0),
        ):
            doc=copy.deepcopy(self.doc)
            mutate(doc)
            self.assertEqual(anchor_entity_region_ids(doc), ())
            self.assertIsNone(compile_entity_anchor(doc, self.geometry))

    def test_scaled_anchor_with_different_patch_aligned_aspect(self):
        from coherent_diffusion.scheduler.sdxl_sync import TilePlan
        from coherent_diffusion.scheduler.regional import compile_entity_anchor
        # Rounded DiT patch alignment need not preserve the target aspect.
        anchor=TilePlan.build(512,320,512,0,0,scale=8)
        out=compile_entity_anchor(self.doc,anchor)
        self.assertEqual(len(out.states),2)
        self.assertGreater(float(next(s for s in out.states
                                      if "small bright sun" in s.state.prompt
                                      ).weight[6,49]),.5)

    def test_anchor_predict_uses_masked_states_not_global_only(self):
        from coherent_diffusion.forge_bridge.regional import _RegionalRunner
        from coherent_diffusion.forge_bridge.anima import ForgeAnima
        from coherent_diffusion.scheduler.regional import PredictionState
        events=[]
        class FakeAnima:
            def __init__(self, prompt):
                self.positive=prompt
                self.negative="negative"
                self.prompt=prompt
            def prepare(self): events.append(("prepare",self.prompt))
            def cleanup(self): events.append(("cleanup",self.prompt))
            def predict(self, crop, sigma, tile):
                return torch.full_like(crop, .9 if "small bright sun" in self.positive else .1)
        runner=_RegionalRunner(
            types.SimpleNamespace(), types.SimpleNamespace(
                mode="progressive",target=self.geometry, anchor=self.geometry),
            self.doc, model="anima",cfg=1.0,seed=1,device="cpu",
            asset_bytes={},loras=types.SimpleNamespace(scoped=lambda *a:None),
            cancelled=None,max_forwards_per_window=None,guidance=())
        self.assertEqual(len(runner.anchor_regional.states),2)
        crop=torch.zeros((1,16,1,self.geometry.height,self.geometry.width))
        with tempfile.TemporaryDirectory() as folder, \
             patch("coherent_diffusion.forge_bridge.diagnostics._data_dir",
                   return_value=Path(folder)), \
             patch.object(ForgeAnima,"from_prompts",
                          side_effect=lambda host, geometry, **kwargs:
                          FakeAnima(kwargs["prompt"])):
            try:
                actual=runner.predict(crop,.5,self.geometry.windows[0],"anchor")
                self.assertGreater(float(actual[0,0,0,8,49]),.65)
                self.assertAlmostEqual(float(actual[0,0,0,8,3]),.1,places=5)
                # Same loaded UNet, two prompt-only conditions, one prepare.
                self.assertEqual(len([name for name,_ in events if name=="prepare"]),1)
            finally:
                runner.close()
        self.assertEqual(len([name for name,_ in events if name=="cleanup"]),1)


if __name__=="__main__":
    unittest.main()
