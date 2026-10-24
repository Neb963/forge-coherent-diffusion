"""P01 deterministic tests: actual CRF and CPU computations, no GPU required."""
import copy
import importlib.util
import json
import math
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

from coherent_diffusion.crf import CRFError, canonical_json, parse_crf, parse_validate, validate_crf
from coherent_diffusion.crf.raster import rasterize
from coherent_diffusion.core.sync_reference import plan_windows, synchronized_step
from coherent_diffusion.bench.metadata import BenchmarkRecord
from coherent_diffusion.forge_bridge.probe import inspect_process, inspect_loaded_host

HERE = Path(__file__).resolve().parent
CASES = json.loads((HERE / "fixtures" / "crf_conformance.json").read_text(encoding="utf-8"))

class CRFConformanceTests(unittest.TestCase):
    def test_fixture_counts(self):
        self.assertGreaterEqual(len(CASES["valid"]),10)
        self.assertGreaterEqual(len(CASES["invalid"]),20)

    def test_all_valid_and_roundtrip(self):
        for case in CASES["valid"]:
            with self.subTest(case=case["name"]):
                doc=case["document"]
                report=validate_crf(doc)
                self.assertIs(report.document,doc)
                roundtrip=parse_validate(canonical_json(doc)).document
                self.assertEqual(doc,roundtrip)
                self.assertEqual(canonical_json(doc),canonical_json(roundtrip))

    def test_all_invalid(self):
        for case in CASES["invalid"]:
            with self.subTest(case=case["name"]):
                with self.assertRaises(CRFError):
                    validate_crf(case["document"])

    def test_example_sample(self):
        sample = json.loads((HERE.parent/"examples/one_sun.crf.json").read_text())
        self.assertEqual(validate_crf(sample).document, sample)
        self.assertEqual(sample["entities"][0]["count"], 1)
        self.assertEqual(sample["entities"][0]["region_ids"], ["sun"])

    def test_schema_meta_valid(self):
        from jsonschema import Draft202012Validator
        schema=json.loads((HERE.parent/"coherent_diffusion/crf/schema_v1.json").read_text())
        Draft202012Validator.check_schema(schema)

    def test_strict_single_fence(self):
        doc=CASES["valid"][0]["document"]
        raw=canonical_json(doc)
        self.assertEqual(parse_crf("  " + raw),doc)
        self.assertEqual(parse_crf(chr(96)*3+"json\n"+raw+"\n"+chr(96)*3),doc)
        for bad in ('{"x":NaN}', '{"a":1,"a":2}', "prefix"+raw,
                    chr(96)*3+"json\n"+raw+"\n"+chr(96)*3+"\nhello",
                    "[]","{}",'{"format": "coherent-regions", "version": 2}'):
            with self.subTest(source=bad[:30]):
                if bad in ("{}", '{"format": "coherent-regions", "version": 2}'):
                    with self.assertRaises(CRFError): parse_validate(bad)
                else:
                    with self.assertRaises(CRFError): parse_crf(bad)

    def test_prompt_absent_null_and_empty_preserved(self):
        docs=[CASES["valid"][i]["document"] for i in (0,3,4)]
        for d in docs:
            self.assertEqual(parse_validate(canonical_json(d)).document,d)
        self.assertNotIn("prompt",docs[2]["regions"][0])
        self.assertIsNone(docs[2]["regions"][0]["negative_prompt"])
        self.assertEqual(docs[1]["regions"][0]["prompt"],"")

    def test_aspect_requires_preview_warning(self):
        d=CASES["valid"][0]["document"]
        self.assertTrue(any("Aspect mismatch" in x for x in validate_crf(d,target_canvas=(1024,1024)).warnings))
        self.assertFalse(any("Aspect mismatch" in x for x in validate_crf(d,target_canvas=(2048,1536)).warnings))

    def test_fuzz_strict_no_crash(self):
        d=CASES["valid"][0]["document"]
        for i in range(100):
            v=copy.deepcopy(d)
            v["regions"][0]["geometry"]["bbox"]=[(i%31)/30,0,.5,1]
            try:
                validate_crf(v)
            except CRFError:
                pass
            else:
                self.assertLessEqual(sum((v["regions"][0]["geometry"]["bbox"][j] for j in (0,2))),1+1e-9)

class RasterTests(unittest.TestCase):
    def test_half_canvas_bbox(self):
        g={"type":"bbox","bbox":[0,0,.5,1]}
        expected=[[1.,1.,0.,0.]]*4
        self.assertEqual(rasterize(g,4,4),expected)

    def test_polygon_clockwise_and_ccw(self):
        p=[[0,0],[1,0],[1,1],[0,1]]
        self.assertEqual(rasterize({"type":"polygon","points":p},7,5),
                         rasterize({"type":"polygon","points":list(reversed(p))},7,5))
        self.assertEqual(sum(sum(row) for row in rasterize({"type":"polygon","points":p},7,5)),35)

    def test_overlap_keeps_independent_masks(self):
        a = rasterize({"type":"bbox","bbox":[0,0,.75,1]},4,2)
        b = rasterize({"type":"bbox","bbox":[.25,0,.75,1]},4,2)
        self.assertEqual(a[0], [1.,1.,1.,0.])
        self.assertEqual(b[0], [0.,1.,1.,1.])
        self.assertEqual(sum(1 for x,y in zip(a[0],b[0]) if x and y), 2)

    def test_reject_unsupported_binary(self):
        with self.assertRaises(CRFError):
            rasterize({"type":"mask","asset_id":"x"},8,8)

class TileTests(unittest.TestCase):
    @staticmethod
    def pred(crop,win):
        assert isinstance(crop,tuple) and isinstance(crop[0],tuple)
        return [[v*.25+(win.x0+win.y0)*.1 for v in row] for row in crop]

    def test_coverage_order_invariance(self):
        old=[[float(x+y*3) for x in range(11)] for y in range(8)]
        old_snapshot=copy.deepcopy(old)
        windows=plan_windows(11,8,5,2)
        def advance(z,p):return z-p*.1
        out=synchronized_step(old,windows,self.pred,advance)
        reversed_out=synchronized_step(old,list(reversed(windows)),self.pred,advance)
        self.assertEqual(out,reversed_out)
        self.assertEqual(old,old_snapshot)
        self.assertEqual((len(out),len(out[0])),(8,11))

    def test_uncovered_and_shape_rejected(self):
        state=[[1.]*4 for _ in range(4)]
        with self.assertRaises(ValueError):
            synchronized_step(state,plan_windows(2,2,2,0),self.pred,lambda z,p:z-p)
        with self.assertRaises(ValueError):
            synchronized_step(state,plan_windows(4,4,2,1),lambda crop,w:[[0.]],lambda z,p:z-p)
        with self.assertRaises(ValueError):
            plan_windows(4,4,2,2)

    def test_constant_prediction(self):
        w=plan_windows(5,3,3,1)
        out=synchronized_step([[2.]*5 for _ in range(3)],w,
            lambda crop,win:[[1.]*len(r) for r in crop],
            lambda z,p:z-p)
        self.assertEqual(out,[[1.]*5 for _ in range(3)])

class EvidenceTests(unittest.TestCase):
    def test_metadata_status_and_serialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/"bench.jsonl"
            a=BenchmarkRecord(model_family="Anima",mode="P01-probe")
            a.save(dest)
            self.assertEqual(json.loads(dest.read_text().splitlines()[0])["status"],"not_tested")
            with self.assertRaises(ValueError):
                BenchmarkRecord("SDXL","not_run",measurements={"seconds":5.}).save(dest)
            b=BenchmarkRecord("SDXL","baseline",status="pass",measurements={"elapsed_s":1.3})
            b.save(dest)
            self.assertEqual(len(dest.read_text().splitlines()),2)

    def test_probe_no_model_and_fake_model(self):
        p=types.SimpleNamespace(sd_model=None,steps=20,width=1024,height=1024)
        result=inspect_process(p)
        self.assertFalse(result["model_loaded"])
        self.assertEqual(result["capabilities"]["q4_gguf_anima_encode"],"not_tested")
        model=types.SimpleNamespace(is_sdxl=True,is_wan=False,model_config=None,
                forge_objects=types.SimpleNamespace(unet=None,clip=None,vae=None))
        p.sd_model=model
        self.assertEqual(inspect_process(p)["model_family"],"SDXL (host flag)")
        self.assertFalse(inspect_process(p)["text_encoder"]["gguf_parameter_marker_seen"])

    def test_script_api_shape_with_fake_host(self):
        class Context:
            def __enter__(self):return self
            def __exit__(self,*args):return False
        class Button:
            def click(self,**kwargs):
                self.func=kwargs["fn"]
        gr=types.ModuleType("gradio")
        gr.Accordion=lambda *a,**k:Context()
        gr.Markdown=lambda *a,**k:None
        gr.Checkbox=lambda **k:True
        gr.Button=lambda *a,**k:Button()
        gr.JSON=lambda **k:None
        gr.HTML=lambda **k:None
        mod=types.ModuleType("modules")
        scr=types.ModuleType("modules.scripts")
        scr.Script=type("Script",(object,),{})
        scr.AlwaysVisible=object()
        mod.scripts=scr
        with mock.patch.dict(sys.modules,{"gradio":gr,"modules":mod,"modules.scripts":scr}):
            spec=importlib.util.spec_from_file_location("cd_probe_script",HERE.parent/"scripts/coherent_diffusion.py")
            instance=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(instance)
            script=instance.Script()
            self.assertIs(script.show(False),scr.AlwaysVisible)
            self.assertFalse(script.show(True))
            self.assertEqual(script.ui(False),[True])
            script.process(types.SimpleNamespace(sd_model=None),True)
            self.assertFalse(instance.probe_ui()["model_loaded"])

if __name__ == "__main__": unittest.main()
