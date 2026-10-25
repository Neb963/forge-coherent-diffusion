"""P07 live Anima hook-failure triage: offline-only unit regression."""
import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:
    torch = None

from coherent_diffusion.forge_bridge.probe import _inspect, _unet_hooks


class ProbeHookDiagnostics(unittest.TestCase):
    def test_reports_only_hook_metadata(self):
        u = SimpleNamespace(controlnet_linked_list=object(),
                            extra_concat_condition=None,
                            model_options={"transformer_options": {"patches": [object()]}})
        d = _unet_hooks(u)
        self.assertTrue(d["controlnet_attached"])
        self.assertFalse(d["extra_concat_condition_attached"])
        self.assertEqual(d["model_option_keys"], ["transformer_options"])
        self.assertEqual(d["transformer_option_keys"], ["patches"])
        self.assertNotIn("value", str(d))
        self.assertEqual(_unet_hooks(None), {"available": False})

    def test_placeholder_is_not_marked_as_loaded(self):
        FakeInitialModel = type("FakeInitialModel", (), {"__module__": "modules.sd_models"})
        status = _inspect(FakeInitialModel(), source="offline")
        self.assertFalse(status["model_loaded"])
        self.assertFalse(status["unet_hooks"]["available"])


@unittest.skipUnless(torch is not None, "Anima adapter requires torch")
class HookRejection(unittest.TestCase):
    def test_which_hook_is_identified(self):
        from coherent_diffusion.forge_bridge.anima import _check_unet_hooks, UnsupportedAnima
        def patcher(control=None, concat=None, opts=None):
            return SimpleNamespace(controlnet_linked_list=control,
                                   extra_concat_condition=concat,
                                   model_options={} if opts is None else opts)
        _check_unet_hooks(patcher())
        _check_unet_hooks(patcher(opts={"transformer_options": {}}))
        cases = [
            (patcher(control=object()), "ControlNet attachment"),
            (patcher(concat=object()), "extra_concat_condition"),
            (patcher(opts={"patches": {}}), "model_options keys"),
            (patcher(opts={"transformer_options": {"patches": []}}), "transformer_option_keys"),
            (patcher(opts=42), "not a dict"),
        ]
        for obj, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(UnsupportedAnima, expected):
                    _check_unet_hooks(obj)
        self.assertEqual(patcher().model_options, {})

    def test_native_sampler_sigma_bookkeeping_only(self):
        import torch
        from coherent_diffusion.forge_bridge.anima import _check_unet_hooks, UnsupportedAnima
        def obj(value, extra=None):
            nested = {"sampling_sigmas": value}
            if extra is not None:
                nested["patches"] = extra
            return SimpleNamespace(controlnet_linked_list=None,
                                   extra_concat_condition=None,
                                   model_options={"transformer_options": nested})
        _check_unet_hooks(obj(torch.tensor([1.0, 0.7, 0.0])))
        _check_unet_hooks(obj([1.0, 0.5, 0.0]))
        _check_unet_hooks(obj((1.0, 0.5, 0.0)))
        invalid = [None, [], [0.5], [0.3, float("nan")],
                   torch.tensor([0.5, float("inf")]),
                   torch.zeros(1, 2), torch.tensor([1, 0]),
                   torch.zeros(1025)]
        for value in invalid:
            with self.subTest(type=type(value).__name__):
                with self.assertRaisesRegex(UnsupportedAnima, "sampling_sigmas has an invalid"):
                    _check_unet_hooks(obj(value))
        with self.assertRaisesRegex(UnsupportedAnima, "Unnegotiated"):
            _check_unet_hooks(obj([1.0, 0.0], extra=[object()]))
        with self.assertRaisesRegex(UnsupportedAnima, "Unnegotiated"):
            u = obj([1.0, 0.0])
            u.model_options["sampler_post_cfg_function"] = [lambda x: x]
            _check_unet_hooks(u)



if __name__ == "__main__":
    unittest.main()
