"""P07 real work tracing and model-residency regression (offline, no GPU)."""
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch=None

from coherent_diffusion.forge_bridge import diagnostics as d


class TimingJournal(unittest.TestCase):
    def test_progress_marks_start_before_blocking_prediction_and_tracks_failure(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(d,"_data_dir",return_value=Path(folder)):
            d.start_progress(upper_bound=1200,window_total=20,region_states=6)
            try:
                self.assertIn("at most 1200",d.status_text())
                with d.timed_prediction(stage="anchor",tile=0,sigma=1.0):
                    self.assertIn("RUNNING",d.status_text())
                    self.assertIn("prediction_started",d.status_text())
                self.assertIn("1 finished",d.status_text())
                with self.assertRaisesRegex(RuntimeError,"simulated stall failure"):
                    with d.timed_prediction(stage="refine",tile=3,sigma=.5):
                        raise RuntimeError("simulated stall failure")
                self.assertIn("1 finished",d.status_text())
                text=d.recent_text(30)
                self.assertIn('"event":"prediction_begin"',text)
                self.assertIn('"event":"prediction_end"',text)
                self.assertIn('"event":"prediction_failure"',text)
                self.assertIn('"tile":3',text)
            finally:
                d.finish_progress(success=False)
            self.assertIn("STOPPED_OR_ERROR",d.status_text())


@unittest.skipIf(torch is None,"Forge regional Tensor modules need Torch")
class ResidentAnimaPromptStates(unittest.TestCase):
    def test_prompt_only_states_reuse_prepared_unet_and_restore_per_state_condition(self):
        from coherent_diffusion.forge_bridge.regional import _RegionalRunner
        from coherent_diffusion.forge_bridge.anima import ForgeAnima
        from coherent_diffusion.scheduler.regional import PredictionState

        operations=[]
        class FakeAdapter:
            def __init__(self,label):
                self.positive=label
                self.negative="negative-"+label
                self.ready=False
                self.plan=None
            def prepare(self):
                self.ready=True
                operations.append(("prepare",self.positive))
            def cleanup(self):
                self.ready=False
                operations.append(("cleanup",self.positive))
            def predict(self,crop,sigma,tile):
                self.test_cond = (self.positive,self.negative)
                return self.test_cond

        runner=_RegionalRunner.__new__(_RegionalRunner)
        runner.model="anima"
        runner.host=object()
        runner.cfg=1.0
        runner.seed=1
        runner.device="cpu"
        runner.cached={}
        runner.condition_cache={}
        runner.active=None
        runner.active_scope=None
        runner.active_key=None
        runner.cancelled=None
        runner.guidance=types.SimpleNamespace(requests=[])
        runner.quality=types.SimpleNamespace(
            target=object(),anchor=object())
        runner.loras=types.SimpleNamespace(scoped=lambda *args: None)
        state_a=PredictionState("A","",())
        state_b=PredictionState("B","",())
        tile=types.SimpleNamespace(index=2)
        original=ForgeAnima.from_prompts
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(d,"_data_dir",return_value=Path(folder)), \
             patch.object(ForgeAnima,"from_prompts",
                          side_effect=lambda *a,**kw:FakeAdapter(kw["prompt"])):
            try:
                output=[runner.predict_state(s,None,.5,tile,"refine")
                        for s in (state_a,state_b,state_a,state_b)]
                self.assertEqual(output,[("A","negative-A"),("B","negative-B"),
                                         ("A","negative-A"),("B","negative-B")])
                self.assertEqual(operations,[("prepare","A")],
                                 "No repeated sampling_cleanup/prepare on unchanged Anima UNet")
                runner.predict_state(state_a,None,.5,tile,"anchor")
                self.assertEqual([op[0] for op in operations],
                                 ["prepare","cleanup","prepare"])
            finally:
                runner.close()
        self.assertEqual([op[0] for op in operations],
                         ["prepare","cleanup","prepare","cleanup"])


if __name__=="__main__":
    unittest.main()
