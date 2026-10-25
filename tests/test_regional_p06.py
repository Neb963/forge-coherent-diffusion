"""P06 CPU-only numerical CRF/regional/Forge-state contract tests."""
import hashlib
import io
import sys
import tempfile
import types
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


def scene():
    return dict(
        format="coherent-regions", version=1,
        canvas=dict(width=128, height=96,
                    coordinate_system="normalized_top_left",
                    aspect_policy="preserve_relative"),
        global_={"scene_prompt": "one lighthouse", "style_prompt": "cinematic",
                 "negative_prompt": "blur", "base_weight": 1},
        blend={"policy": "weighted"}, regions=[])


def fixture():
    d = scene()
    d["global"] = d.pop("global_")
    return d


def region(id, bbox, *, weight=1, z=0, prompt="region", feather=0):
    return dict(id=id, geometry={"type": "bbox", "bbox": bbox}, prompt=prompt,
                blend={"weight": weight, "z_index": z, "feather_px": feather})


@unittest.skipIf(torch is None, "P06 uses pinned CPU torch tests")
class RegionalTests(unittest.TestCase):
    def setUp(self):
        from coherent_diffusion.scheduler.sdxl_sync import TilePlan
        self.tiles = TilePlan.build(128, 96, 64, 16, 8)

    def compile(self, doc, **kwargs):
        from coherent_diffusion.scheduler.regional import compile_regional
        return compile_regional(doc, self.tiles, **kwargs)

    def test_weighted_overlap_and_shared_style(self):
        d = fixture()
        d["regions"] = [
            region("left", [0, 0, .75, 1], weight=3, prompt="light"),
            region("right", [.25, 0, .75, 1], weight=2, prompt="dark")]
        p = self.compile(d)
        self.assertEqual(len(p.states), 3)
        shares = {s.state.prompt: s.weight for s in p.states}
        self.assertAlmostEqual(shares["cinematic"][5, 8].item(), 1/6, places=5)
        self.assertAlmostEqual(shares["cinematic, light"][5, 8].item(), 3/6, places=5)
        self.assertAlmostEqual(shares["cinematic, dark"][5, 8].item(), 2/6, places=5)
        self.assertTrue(torch.allclose(sum(shares.values()), torch.ones_like(next(iter(shares.values())))))
        self.assertNotIn("lighthouse", " ".join(shares))

    def test_priority_fractional_feather_and_stable_ties(self):
        d = fixture()
        d["blend"]["policy"] = "priority"
        d["regions"] = [region("back", [0, 0, 1, 1], z=2, prompt="back"),
                        region("front", [.25, 0, .5, 1], z=2, prompt="front", feather=24)]
        p = self.compile(d)
        by_name = {s.state.prompt: s.weight for s in p.states}
        center = (5, 8)
        self.assertGreater(by_name["cinematic, front"][center].item(),
                           by_name["cinematic, back"][center].item())
        self.assertTrue(torch.allclose(sum(by_name.values()), torch.ones_like(next(iter(by_name.values())))))
        # At left boundary of the feather band, the front cannot instantly
        # erase the back; stable z/document order remains global.
        self.assertGreater(by_name["cinematic, back"][5, 4].item(), 0)
        reverse = deepcopy(d)
        reverse["regions"].reverse()
        r = self.compile(reverse)
        by_name2 = {s.state.prompt: s.weight for s in r.states}
        self.assertFalse(torch.equal(by_name["cinematic, front"], by_name2.get("cinematic, front", torch.zeros_like(by_name["cinematic, front"]))))

    def test_disabled_visibility_zero_and_negative_semantics(self):
        d = fixture()
        a = region("r", [0, 0, 1, 1], prompt="")
        a["visible"] = False
        d["regions"] = [a]
        p = self.compile(d)
        self.assertEqual(len(p.states), 1)  # empty prompt inherits style, same state as base
        a["prompt"] = None
        self.assertEqual(len(self.compile(d).states), 1)
        a["prompt"] = "forest"
        self.assertEqual(len(self.compile(d).states), 2)
        a["enabled"] = False
        self.assertEqual(len(self.compile(d).states), 1)
        a["enabled"] = True
        a["blend"]["weight"] = 0
        self.assertEqual(len(self.compile(d).states), 1)
        a["blend"]["weight"] = 1
        a["negative_prompt"] = ""
        p = self.compile(d)
        self.assertEqual({s.state.negative for s in p.states}, {"blur", ""})

    def test_polygon_and_invalid_aspect(self):
        from coherent_diffusion.scheduler.regional import UnsupportedRegional
        d = fixture()
        d["regions"] = [dict(id="poly", geometry={"type": "polygon",
                             "points": [[.1, .1], [.9, .1], [.5, .9]]},
                             blend={"weight": 2, "z_index": 0, "feather_px": 8},
                             prompt="triangle")]
        p = self.compile(d)
        self.assertGreater(max(s.weight[5, 8].item() for s in p.states), 0)
        d["canvas"]["width"] = 256
        with self.assertRaises(UnsupportedRegional):
            self.compile(d)

    def test_hash_verified_png_mask_and_corruption(self):
        from PIL import Image
        from coherent_diffusion.scheduler.regional import UnsupportedRegional
        img = Image.new("RGBA", (128, 96), (30, 20, 10, 0))
        for y in range(96):
            for x in range(64):
                img.putpixel((x, y), (0, 0, 0, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw = buf.getvalue()
        d = fixture()
        d["assets"] = [dict(id="img", path="masks/img.png",
                            sha256=hashlib.sha256(raw).hexdigest(),
                            media_type="image/png", width=128, height=96)]
        d["regions"] = [dict(id="r", geometry={"type": "mask",
                                "asset_id": "img", "interpolation": "nearest"},
                             blend={"weight": 1, "z_index": 0, "feather_px": 0},
                             prompt="ink")]
        with self.assertRaises(UnsupportedRegional):
            self.compile(d)
        p = self.compile(d, asset_bytes={"img": raw})
        self.assertAlmostEqual(p.states[1].weight[5, 2].item(), .5, places=6)
        self.assertEqual(p.states[1].weight[5, 14].item(), 0)
        with self.assertRaises(UnsupportedRegional):
            self.compile(d, asset_bytes={"img": raw + b"tamper"})
        d["regions"][0]["blend"]["feather_px"] = 8
        with self.assertRaises(UnsupportedRegional):
            self.compile(d, asset_bytes={"img": raw})

    def test_identical_prediction_group_and_window_read_coordinates(self):
        from coherent_diffusion.scheduler.regional import blended_prediction
        d = fixture()
        d["regions"] = [region("a", [0, 0, .8, 1], prompt="same"),
                        region("b", [.2, 0, .8, 1], prompt="same")]
        p = self.compile(d)
        self.assertEqual(len(p.states), 2)
        t = self.tiles.windows[-1]
        for dims in (4, 5):
            shape = (1, 4, t.read.height, t.read.width) if dims == 4 else (
                1, 16, 1, t.read.height, t.read.width)
            crop = torch.zeros(shape)
            seen = []
            def predict(state, same_crop, sigma, tile, stage):
                seen.append((state.prompt, sigma, tile.index, stage))
                return torch.full_like(same_crop, 3 if "same" in state.prompt else 1)
            blended = blended_prediction(p, t, crop, .5, "direct", predict)
            self.assertEqual(blended.shape, crop.shape)
            self.assertEqual(len(seen), len(p.window(t)))
            coeff = {s.state.prompt: s.weight for s in p.states}
            weights = coeff["cinematic, same"][t.read.y0:t.read.y1, t.read.x0:t.read.x1]
            self.assertTrue(torch.allclose(blended[0, 0].squeeze(), 1 + weights*2))
            with self.assertRaises(UnsupportedRegionalError()):
                blended_prediction(p, t, crop, .5, "anchor", predict)

    def test_unsafe_lora_intents_and_groups(self):
        from coherent_diffusion.scheduler.regional import UnsupportedRegional
        d = fixture()
        r = region("r", [0, 0, 1, 1])
        r["loras"] = [dict(id="l", name="look", weight=.7, scope="regional",
                           requested_isolation="prediction", te_weight=.5)]
        d["regions"] = [r]
        with self.assertRaises(UnsupportedRegional):
            self.compile(d)
        r["loras"][0]["te_weight"] = None
        p = self.compile(d)
        self.assertEqual(p.states[-1].state.adapters, (("look", None, .7),))
        r["loras"][0]["requested_isolation"] = "experimental_internal"
        with self.assertRaises(UnsupportedRegional):
            self.compile(d)

    def test_native_lora_state_scope_restoration_and_sha(self):
        from coherent_diffusion.forge_bridge.regional import NativeForgeLoRAs
        from coherent_diffusion.scheduler.regional import UnsupportedRegional
        with tempfile.TemporaryDirectory() as td:
            file = Path(td) / "trusted.safetensors"
            file.write_bytes(b"source-backed registry mock")
            net = types.ModuleType("fake_networks")
            net.available_network_aliases = {"look": SimpleNamespace(filename=str(file))}
            net.load_lora_state_dict = lambda name: {"k": b"v"}
            net.load_lora_for_models = lambda model, clip, state, weight, te, **kw: (
                SimpleNamespace(model=object()), None)
            backend = types.ModuleType("backend")
            backend.args = types.ModuleType("backend.args")
            backend.args.dynamic_args = SimpleNamespace(nunchaku=False)
            base_unet, clip = object(), object()
            host = SimpleNamespace(forge_objects=SimpleNamespace(unet=base_unet, clip=clip))
            with patch.dict(sys.modules, {"fake_networks": net, "backend": backend,
                                          "backend.args": backend.args}):
                manager = NativeForgeLoRAs("fake_networks")
                sha = hashlib.sha256(file.read_bytes()).hexdigest()
                with manager.scoped(host, (("look", sha, .7),)):
                    self.assertIsNot(host.forge_objects.unet, base_unet)
                    self.assertIs(host.forge_objects.clip, clip)
                    with self.assertRaises(UnsupportedRegional):
                        with manager.scoped(host, (("look", None, .7),)):
                            pass
                self.assertIs(host.forge_objects.unet, base_unet)
                self.assertIs(host.forge_objects.clip, clip)
                with self.assertRaises(UnsupportedRegional):
                    with manager.scoped(host, (("look", "0"*64, .7),)):
                        pass
                self.assertIs(host.forge_objects.unet, base_unet)


def UnsupportedRegionalError():
    from coherent_diffusion.scheduler.regional import UnsupportedRegional
    return UnsupportedRegional


@unittest.skipIf(torch is None, "P06 uses pinned CPU torch tests")
class ForgeWiringTests(unittest.TestCase):
    def test_sdxl_and_anima_stage_integration_cleanup_and_global_anchor(self):
        from coherent_diffusion.scheduler.sdxl_sync import TilePlan
        from coherent_diffusion.scheduler.quality import QualityPlan
        from coherent_diffusion.scheduler.anima_flow import AnimaQualityPlan
        from coherent_diffusion.forge_bridge import regional as bridge
        p = TilePlan.build(128, 96, 64, 16, 16)
        d = fixture()
        d["regions"] = [region("r", [0, 0, .5, 1], prompt="specific")]
        events = []

        class Fake:
            def __init__(self, prompt):
                self.prompt = prompt
            @classmethod
            def from_prompts(cls, host, plan, **kw):
                events.append(("encode", kw["prompt"], plan.width_px))
                return cls(kw["prompt"])
            def prepare(self):
                events.append(("prepare", self.prompt))
            def predict(self, crop, sigma, tile):
                return torch.zeros_like(crop)  # valid synthetic native x0
            def cleanup(self):
                events.append(("cleanup", self.prompt))
        with patch.object(bridge, "ForgeSDXL", Fake), patch.object(bridge, "ForgeAnima", Fake):
            sd = QualityPlan.build(p, mode="progressive", refine_sigmas=(1, 0),
                                   anchor_sigmas=(1, 0), anchor_factor=2)
            result = bridge.run_forge_regional_sdxl(object(), sd, d, cfg=3, seed=10,
                                                    device="cpu")
            self.assertEqual(result.mode, "progressive")
            self.assertIn(("encode", "one lighthouse, cinematic", 64), events)
            self.assertIn(("encode", "cinematic, specific", 128), events)
            self.assertIn(("encode", "cinematic", 128), events)
            events.clear()
            an = AnimaQualityPlan.build(p, patch_spatial=2, mode="direct",
                                        refine_sigmas=(1, 0))
            result = bridge.run_forge_regional_anima(object(), an, d, seed=10,
                                                     device="cpu", inference_dtype=torch.float32)
            self.assertEqual(result.mode, "direct")
            self.assertNotIn(("encode", "one lighthouse, cinematic", 128), events)
            self.assertIn(("encode", "cinematic, specific", 128), events)
            self.assertGreater(len([e for e in events if e[0] == "cleanup"]), 0)


if __name__ == "__main__":
    unittest.main()
