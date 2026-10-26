"""P04 global-first / direct SDXL and entity-aware routing offline contracts.

No image-quality/GPU assertions: these are controlled synthetic predictions,
numeric geometry, lineage and failure-handling tests only.
"""
import json
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "P04 tensor suite executes in pinned CPU torch CI job")
class QualityP04Tests(unittest.TestCase):
    def setUp(self):
        from coherent_diffusion.scheduler.sdxl_sync import TilePlan
        from coherent_diffusion.scheduler.quality import QualityPlan, execute_quality
        self.TilePlan, self.QualityPlan, self.execute = TilePlan, QualityPlan, execute_quality
        self.plan = TilePlan.build(128, 96, 64, 16, 8)

    def test_direct_determinism_one_global_solver(self):
        q = self.QualityPlan.build(self.plan, mode="direct", refine_sigmas=(2., 1., 0.))
        calls = []
        def f(crop,sigma,tile,stage):
            calls.append((stage,sigma,tile.index))
            return crop * .5
        a = self.execute(q, 12, f)
        b = self.execute(q, 12, f)
        c = self.execute(q, 13, f)
        self.assertTrue(torch.equal(a.latent,b.latent))
        self.assertFalse(torch.equal(a.latent,c.latent))
        self.assertEqual(a.latent.device.type,"cpu")
        self.assertEqual(len(a.stages),1)
        self.assertEqual(a.stages[0].name,"direct")
        self.assertEqual(len(a.stages[0].steps),2)
        self.assertEqual(len(calls),3*2*len(self.plan.windows))
        self.assertEqual(a.model_forwards,2*len(self.plan.windows))
        self.assertTrue(bool(torch.isfinite(a.latent).all()))

    def test_progressive_real_anchor_then_global_refinement(self):
        q = self.QualityPlan.build(self.plan, mode="progressive",
                                   anchor_sigmas=(1.,0.), refine_sigmas=(.5,0.),
                                   anchor_factor=2)
        seen = []
        def f(crop,sigma,tile,stage):
            seen.append((stage,crop.shape[-2:],tile.index))
            if stage=="anchor":
                return torch.ones_like(crop) * .25
            return crop * .5
        got = self.execute(q, 7, f)
        self.assertEqual([s.name for s in got.stages], ["anchor","refine"])
        self.assertEqual(len(q.anchor.windows),1)
        self.assertEqual(got.stages[0].geometry.width,self.plan.width//2)
        self.assertEqual(tuple(got.latent.shape),(1,4,self.plan.height,self.plan.width))
        self.assertEqual(got.model_forwards,1+len(self.plan.windows))
        self.assertEqual(len(seen),got.model_forwards)
        self.assertEqual(seen[0][0],"anchor")
        self.assertTrue(all(name=="refine" for name,_,_ in seen[1:]))
        from coherent_diffusion.scheduler.sdxl_sync import seeded_noise
        expected = (.25+seeded_noise(self.plan, 7 ^ 0x9E3779B97F4A7C15,
                                       sigma=.5)) * .5
        self.assertTrue(torch.allclose(got.latent,expected,atol=1e-6))
        self.assertTrue(torch.equal(got.latent,self.execute(q,7,f).latent))
        self.assertFalse(torch.equal(got.latent,self.execute(q,8,f).latent))


    def test_toy_local_repetition_vs_one_global_anchor_multiseed(self):
        """Toy denoiser comparison only: not a diffusion model or quality proof."""
        def blobs(tensor):
            pixels=(tensor[0,0] > .40).tolist()
            seen=set()
            count=0
            for y in range(len(pixels)):
                for x in range(len(pixels[0])):
                    if not pixels[y][x] or (x,y) in seen:
                        continue
                    count+=1
                    stack=[(x,y)]
                    while stack:
                        px,py=stack.pop()
                        if (px,py) in seen or not 0<=py<len(pixels) or not 0<=px<len(pixels[0]):
                            continue
                        if not pixels[py][px]:
                            continue
                        seen.add((px,py))
                        stack.extend([(px-1,py),(px+1,py),(px,py-1),(px,py+1)])
            return count

        from coherent_diffusion.scheduler.quality import QualityPlan
        direct=QualityPlan.build(self.plan,mode="direct",refine_sigmas=[1,0])
        progressive=QualityPlan.build(self.plan,mode="progressive",
                                      anchor_sigmas=[1,0],refine_sigmas=[.001,0])
        def predictor(crop,sigma,tile,stage):
            yy,xx=torch.meshgrid(torch.arange(crop.shape[-2]),
                                 torch.arange(crop.shape[-1]),indexing="ij")
            if stage=="anchor":
                # Single globally established entity near the top-right.
                mx,my=5.,1.5
            elif stage=="refine":
                return crop
            else:
                # Bad control: each window receives a complete scene and
                # independently emits its own local entity at tile center.
                mx=(tile.write.x0+tile.write.x1)/2-tile.read.x0
                my=(tile.write.y0+tile.write.y1)/2-tile.read.y0
            gaussian=torch.exp(-((xx-mx)**2+(yy-my)**2)/(2*.8**2))
            return gaussian[None,None].expand_as(crop).clone()
        for seed in (11,29,57):
            with self.subTest(seed=seed):
                local=self.execute(direct,seed,predictor).latent
                global_first=self.execute(progressive,seed,predictor).latent
                self.assertGreater(blobs(local),1)
                self.assertEqual(blobs(global_first),1)
                self.assertTrue(torch.equal(
                    global_first,self.execute(progressive,seed,predictor).latent))

    def test_stage_progress_lineage_and_cleanup(self):
        q = self.QualityPlan.build(self.plan, mode="progressive",
                                   anchor_sigmas=(2,1,0),refine_sigmas=(1,0))
        events, cleanup = [], []
        result = self.execute(q,1,lambda crop,s,t,n: crop,
                              progress=lambda name,info: events.append((name,info.index)),
                              cleanup=lambda: cleanup.append(True))
        self.assertEqual(events,[("anchor",0),("anchor",1),("refine",0)])
        self.assertEqual(len(cleanup),1)
        self.assertEqual([len(s.steps) for s in result.stages],[2,1])

    def test_fail_closed_modes_schedules_geometry(self):
        build=self.QualityPlan.build
        for kwargs in (
            dict(mode="progressive",refine_sigmas=[1,0]),
            dict(mode="progressive",refine_sigmas=[1,0],anchor_sigmas=[2,2,0]),
            dict(mode="direct",refine_sigmas=[1,0],anchor_sigmas=[1,0]),
            dict(mode="direct",refine_sigmas=[1,0],global_interaction="FFT"),
            dict(mode="quality",refine_sigmas=[1,0]),
            dict(mode="direct",refine_sigmas=[0,0]),
            dict(mode="progressive",refine_sigmas=[1,0],anchor_sigmas=[1,0],anchor_factor=3),
        ):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                build(self.plan,**kwargs)
        with self.assertRaises(MemoryError):
            build(self.plan,mode="progressive",refine_sigmas=[1,0],
                  anchor_sigmas=[1,0],max_anchor_pixels=1)
        # P07: output dimensions no longer require an exact factor divisor.
        # Align a full-canvas overview DOWN to the model's latent grid.
        unusual=self.TilePlan.build(120, 96, 64, 16, 8)
        adaptive=build(unusual,mode="progressive",
                       refine_sigmas=[1,0],anchor_sigmas=[1,0])
        self.assertEqual(adaptive.anchor.width_px % unusual.scale,0)
        self.assertEqual(adaptive.anchor.height_px % unusual.scale,0)
        self.assertLess(adaptive.anchor.width_px,unusual.width_px)

    def test_two_stage_preflight_blocks_model_before_anchor(self):
        q=self.QualityPlan.build(self.plan,mode="progressive",
                                 anchor_sigmas=[1,0],refine_sigmas=[1,0])
        calls=[]
        with self.assertRaises(MemoryError):
            self.execute(q,1,lambda *args: calls.append(True),
                         max_window_bytes=1,cleanup=lambda: calls.append("clean"))
        self.assertEqual(calls,["clean"])

    def test_interruption_in_anchor_and_refine_failure_cleanup(self):
        q=self.QualityPlan.build(self.plan,mode="progressive",
                                 anchor_sigmas=[1,0],refine_sigmas=[1,0])
        for failing in ("anchor","refine"):
            seen=[]
            def f(crop,s,t,stage):
                seen.append(stage)
                if stage==failing:
                    raise RuntimeError("inference interrupted")
                return crop
            clean=[]
            with self.assertRaisesRegex(RuntimeError,"interrupted"):
                self.execute(q,1,f,cleanup=lambda: clean.append(True))
            self.assertEqual(clean,[True])
            self.assertIn(failing,seen)
        clean=[]
        with self.assertRaises(InterruptedError):
            self.execute(q,1,lambda *x: self.fail("predict called"),
                         cancelled=lambda: True,cleanup=lambda: clean.append(True))
        self.assertEqual(clean,[True])

    def test_tampered_immutable_quality_plans_fail_without_model_calls(self):
        from dataclasses import replace
        q=self.QualityPlan.build(self.plan,mode="progressive",
                                 anchor_sigmas=[1,0],refine_sigmas=[1,0])
        calls=[]
        invalid=[
            replace(q,mode="unsupported"),
            replace(q,global_interaction="unverified-fft"),
            replace(q,anchor=None),
            replace(q,anchor_sigmas=[1,1,0]),
            replace(q,anchor_factor=4),
            replace(q,anchor=replace(q.anchor,windows=())),
        ]
        direct=self.QualityPlan.build(self.plan,mode="direct",refine_sigmas=[1,0])
        invalid.append(replace(direct,anchor=q.anchor))
        for candidate in invalid:
            with self.subTest(candidate=candidate.mode):
                with self.assertRaises(ValueError):
                    self.execute(candidate,2,lambda *args: calls.append(True))
        with self.assertRaises(ValueError):
            self.execute(direct,-1,lambda *args: calls.append(True))
        self.assertEqual(calls,[])

    def test_anchor_memory_failures_and_nonfinite_model_output(self):
        q=self.QualityPlan.build(self.plan,mode="progressive",
                                 anchor_sigmas=[1,0],refine_sigmas=[1,0])
        clean=[]
        with self.assertRaises(ValueError):
            self.execute(q,3,lambda crop,s,t,stage:
                         torch.full_like(crop,float("nan")),
                         cleanup=lambda: clean.append(True))
        self.assertEqual(clean,[True])


@unittest.skipIf(torch is None, "P04 routing suite executes in pinned CPU torch CI job")
class RoutingP04Tests(unittest.TestCase):
    def setUp(self):
        from coherent_diffusion.scheduler.sdxl_sync import TilePlan
        from coherent_diffusion.scheduler.semantic import compile_routes
        self.plan=TilePlan.build(128,128,64,0,0)
        self.compile=compile_routes
        self.doc={
            "format":"coherent-regions","version":1,
            "canvas":{"width":128,"height":128,
                      "coordinate_system":"normalized_top_left",
                      "aspect_policy":"preserve_relative"},
            "global":{"scene_prompt":"Exactly one sun above two subjects",
                      "style_prompt":"photography, golden light",
                      "negative_prompt":"duplicate objects","base_weight":1},
            "blend":{"policy":"weighted"},
            "regions":[
                {"id":"sun","geometry":{"type":"bbox","bbox":[.35,.1,.3,.3]},
                 "prompt":"one bright sun","visible":False,
                 "blend":{"weight":1,"feather_px":0,"z_index":2}},
                {"id":"subjects","geometry":{"type":"bbox","bbox":[.2,.6,.6,.3]},
                 "prompt":"two people standing together","blend":
                 {"weight":1,"feather_px":0,"z_index":1}},
            ],
            "entities":[
                {"id":"sun-unique","description":"one sun","count":1,
                 "region_ids":["sun"]},
                {"id":"people","description":"pair","count":2,
                 "region_ids":["subjects"]},
            ]
        }

    def test_unique_entity_across_multiple_windows_and_no_global_repetition(self):
        out=self.compile(self.doc,self.plan)
        self.assertFalse(out.connected_to_model)
        self.assertEqual(out.global_scene_prompt,self.doc["global"]["scene_prompt"])
        self.assertIn("photography",out.anchor_prompt)
        self.assertIn("Exactly one sun",out.anchor_prompt)
        self.assertEqual([(e.id,e.count) for e in out.entities],
                         [("sun-unique",1),("people",2)])
        sun_routes=[p for p in out.patches if "sun-unique" in p.entity_ids]
        self.assertEqual(len(sun_routes),2)
        self.assertEqual({p.entity_ids.count("sun-unique") for p in sun_routes},{1})
        self.assertTrue(all("Exactly one sun above" not in p.local_prompt
                            for p in out.patches))
        self.assertTrue(all("photography" in p.local_prompt for p in out.patches))
        self.assertEqual(len(out.patches),len(self.plan.windows))

    def test_visibility_does_not_remove_conditioning_disabled_and_zero_weight_do(self):
        x=self.compile(self.doc,self.plan)
        self.assertTrue(any("sun" in p.region_ids for p in x.patches))
        self.doc["regions"][0]["enabled"]=False
        y=self.compile(self.doc,self.plan)
        self.assertFalse(any("sun" in p.region_ids for p in y.patches))
        self.assertTrue(any("no active spatial" in w for w in y.warnings))
        self.doc["regions"][0]["enabled"]=True
        self.doc["regions"][0]["blend"]["weight"]=0
        z=self.compile(self.doc,self.plan)
        self.assertFalse(any("sun" in p.region_ids for p in z.patches))

    def test_modes_manual_automatic_hybrid_and_proposals(self):
        a=self.compile(self.doc,self.plan,mode="manual")
        self.assertEqual(a.proposals,())
        b=self.compile(self.doc,self.plan,mode="automatic")
        # Automatic mode may use existing uniquely named authored geometry.
        self.assertEqual(b.patches,a.patches)
        self.assertEqual(b.proposals,())
        self.assertTrue(any("pre-authored" in w for w in b.warnings))
        h=self.compile(self.doc,self.plan,mode="hybrid")
        self.assertEqual(h.patches,a.patches)
        self.assertEqual(h.proposals,())
        # An entity named in prose but without matching known coordinates is
        # never invented or injected into arbitrary windows.
        self.doc["global"]["scene_prompt"]="Exactly one phoenix above two subjects"
        auto=self.compile(self.doc,self.plan,mode="automatic")
        self.assertEqual(len(auto.proposals),1)
        self.assertEqual(auto.proposals[0].count,1)
        self.assertFalse(auto.proposals[0].located)
        self.assertFalse(any("sun" in p.region_ids for p in auto.patches))
        self.assertTrue(any("subjects" in p.region_ids for p in auto.patches))

    def test_auto_ambiguous_names_and_conflicting_counts_abstain(self):
        self.doc["global"]["scene_prompt"]="One sun above the sea"
        self.doc["regions"].append(
            {"id":"other","name":"sun","geometry":{"type":"bbox","bbox":[.7,.7,.1,.1]},
             "prompt":"glow","blend":{"weight":1,"feather_px":0,"z_index":0}})
        auto=self.compile(self.doc,self.plan,mode="automatic")
        self.assertEqual(len(auto.proposals),1)
        self.assertFalse(any(p.region_ids for p in auto.patches))
        # No ambiguous shadow entity or fabricated second instance.
        self.doc["regions"].pop()
        self.doc["entities"][0]["count"]=2
        auto=self.compile(self.doc,self.plan,mode="automatic")
        self.assertEqual(auto.entities[0].count,2)
        self.assertTrue(any("conflicts" in w for w in auto.warnings))

    def test_exact_polygon_clipping_not_bounding_box(self):
        self.doc["regions"]=[{"id":"triangle","geometry":{
            "type":"polygon","points":[[0,0],[1,0],[0,1]]},
            "prompt":"triangle","blend":{"weight":1,"feather_px":0,"z_index":1}}]
        self.doc.pop("entities")
        routed=self.compile(self.doc,self.plan)
        self.assertEqual([p.tile_index for p in routed.patches if p.region_ids], [0,1,2])
        # Lower right patch touches diagonal only on a corner, zero area.
        self.assertEqual(routed.patches[3].region_ids,())

    def test_geometry_and_prompt_fields_fail_closed(self):
        with self.assertRaises(ValueError):
            self.compile(self.doc,self.plan,mode="other")
        self.doc["canvas"]["width"]=256
        with self.assertRaises(ValueError):
            self.compile(self.doc,self.plan)
        self.doc["canvas"]["width"]=128
        self.doc["regions"][0]["geometry"]={"type":"bbox","bbox":[.9,.9,.3,.3]}
        with self.assertRaises(Exception):
            self.compile(self.doc,self.plan)

    def test_explicit_prompt_inheritance_null_vs_empty_negative(self):
        self.doc["regions"][0]["negative_prompt"]=""
        self.doc["regions"][1]["negative_prompt"]=None
        result=self.compile(self.doc,self.plan)
        self.assertTrue(any("" in p.local_negative_prompts for p in result.patches))
        self.assertTrue(all(None not in p.local_negative_prompts for p in result.patches))
        self.doc["regions"][0]["inherit_style"]=False
        result=self.compile(self.doc,self.plan)
        self.assertTrue(any("photography" not in p.local_prompt
                            for p in result.patches if p.region_ids == ("sun",)))

    def test_comparison_suite_multiseed_scene_topologies(self):
        """Controlled geometry/identity comparison, not image quality evidence."""
        scenes={
            "one sun":([.4,.05,.2,.3],1),
            "lighthouse":([.46,.15,.1,.65],1),
            "cross-tile bridge":([.2,.4,.6,.2],1),
            "two subjects":([.25,.2,.5,.7],2),
            "architecture":([.08,.15,.84,.8],1),
            "landscape":([0,.45,1,.55],1),
            "perspective":([.42,.3,.16,.7],1),
            "seams":([.45,.05,.1,.9],1),
        }
        for seed in (11,29,57):
            for scene,(bbox,count) in scenes.items():
                with self.subTest(seed=seed,scene=scene):
                    self.doc["global"]["scene_prompt"]=scene
                    self.doc["regions"]=[{"id":"object","geometry":{"type":"bbox","bbox":bbox},
                                          "prompt":scene,"blend":{"weight":1,"feather_px":0,"z_index":0}}]
                    self.doc["entities"]=[{"id":"stable-object","description":scene,
                                            "count":count,"region_ids":["object"]}]
                    route=self.compile(self.doc,self.plan)
                    hits=[p for p in route.patches if "stable-object" in p.entity_ids]
                    self.assertTrue(hits)
                    self.assertEqual(len(route.entities),1)
                    self.assertEqual(route.entities[0].count,count)
                    self.assertTrue(all(p.entity_ids.count("stable-object")==1 for p in hits))
                    self.assertTrue(all(p.region_ids.count("object")==1 for p in hits))
                    self.assertEqual(route,self.compile(self.doc,self.plan))


if __name__=="__main__":
    unittest.main()
