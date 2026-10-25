"""P04 stage-specific Forge bridge lifecycle under source-shaped fake adapter."""
import unittest
from unittest.mock import patch
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "P04 Forge quality stub suite requires CPU torch")
class ForgeQualityP04Tests(unittest.TestCase):
    def setUp(self):
        from coherent_diffusion.scheduler.sdxl_sync import TilePlan
        from coherent_diffusion.scheduler.quality import QualityPlan
        self.plan = TilePlan.build(128, 96, 64, 16, 8)
        self.q = QualityPlan.build(self.plan, mode="progressive",
                                   anchor_sigmas=(1,0), refine_sigmas=(.5,0))

    def make_fake(self, events, fail_stage=None):
        class Fake:
            @classmethod
            def from_prompts(cls, host, plan, *, prompt, negative_prompt, cfg, seed, device):
                stage = "anchor" if plan.width_px == 64 else "target"
                events.append(("encode",stage,plan.width_px,seed,device,prompt,negative_prompt,cfg))
                return cls(stage)
            def __init__(self, stage):
                self.stage=stage
            def prepare(self):
                events.append(("prepare",self.stage))
            def predict(self,crop,sigma,tile):
                events.append(("predict",self.stage,tile.index))
                if fail_stage==self.stage:
                    raise RuntimeError("synthetic Forge failure")
                return crop*.8
            def cleanup(self):
                events.append(("cleanup",self.stage))
        return Fake

    def test_two_stage_metadata_encoder_reuse_and_lifecycle(self):
        from coherent_diffusion.forge_bridge.quality import run_forge_sdxl_quality
        events=[]
        with patch("coherent_diffusion.forge_bridge.quality.ForgeSDXL",self.make_fake(events)):
            r=run_forge_sdxl_quality(object(),self.q,prompt="wide sky",negative_prompt="duplicates",
                                     cfg=7,seed=42,device="cpu")
        self.assertEqual([a[1:3] for a in events if a[0]=="encode"],
                         [("anchor",64),("target",128)])
        self.assertEqual([a for a in events if a[0] in ("prepare","cleanup")],
                         [("prepare","anchor"),("cleanup","anchor"),
                          ("prepare","target"),("cleanup","target")])
        self.assertEqual(len([a for a in events if a[0]=="predict"]),
                         r.model_forwards)
        self.assertTrue(bool(torch.isfinite(r.latent).all()))

    def test_direct_only_prepares_target(self):
        from coherent_diffusion.forge_bridge.quality import run_forge_sdxl_quality
        from coherent_diffusion.scheduler.quality import QualityPlan
        q=QualityPlan.build(self.plan,mode="direct",refine_sigmas=[1,0])
        events=[]
        with patch("coherent_diffusion.forge_bridge.quality.ForgeSDXL",self.make_fake(events)):
            result=run_forge_sdxl_quality(object(),q,prompt="landscape",cfg=1,seed=1,device="cpu")
        self.assertEqual([a[1] for a in events if a[0]=="encode"],["target"])
        self.assertEqual([a[1] for a in events if a[0]=="cleanup"],["target"])
        self.assertEqual(result.mode,"direct")

    def test_inference_failure_cleans_active_stage_and_propagates(self):
        from coherent_diffusion.forge_bridge.quality import run_forge_sdxl_quality
        for stage,expected in [
            ("anchor",["anchor"]),
            ("target",["anchor","target"])
        ]:
            with self.subTest(stage=stage):
                events=[]
                with patch("coherent_diffusion.forge_bridge.quality.ForgeSDXL",
                           self.make_fake(events,fail_stage=stage)):
                    with self.assertRaisesRegex(RuntimeError,"synthetic Forge"):
                        run_forge_sdxl_quality(object(),self.q,prompt="sky",cfg=3,seed=19,device="cpu")
                self.assertEqual([a[1] for a in events if a[0]=="cleanup"],expected)

    def test_no_host_setup_if_cpu_handoff_over_budget(self):
        from coherent_diffusion.forge_bridge.quality import run_forge_sdxl_quality
        events=[]
        with patch("coherent_diffusion.forge_bridge.quality.ForgeSDXL",self.make_fake(events)):
            with self.assertRaises(MemoryError):
                run_forge_sdxl_quality(object(),self.q,prompt="sky",cfg=3,seed=19,
                                        device="cpu",max_cpu_bytes=1024)
        self.assertEqual(events,[])


if __name__=="__main__":
    unittest.main()
