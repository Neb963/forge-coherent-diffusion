"""P07 reproducible installable-extension candidate integrity check."""
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("candidate_builder", ROOT / "scripts" / "build_release_zip.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class PackageTests(unittest.TestCase):
    def test_self_contained_deterministic_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            one=builder.build(Path(temporary) / "first.zip")
            two=builder.build(Path(temporary) / "second.zip")
            self.assertEqual(hashlib.sha256(one.read_bytes()).digest(),
                             hashlib.sha256(two.read_bytes()).digest())
            with ZipFile(one) as archive:
                names=archive.namelist()
                self.assertEqual(len(names),len(set(names)))
                self.assertTrue(all(p.startswith("coherent-diffusion/") for p in names))
                for path in (
                    "scripts/coherent_diffusion.py",
                    "coherent_diffusion/forge_bridge/entrypoint.py",
                    "coherent_diffusion/editor/index.html",
                    "coherent_diffusion/scheduler/anima_flow.py",
                    "coherent_diffusion/forge_bridge/regional.py",
                    "coherent_diffusion/crf/schema_v1.json",
                    "requirements.txt","INSTALL_AND_TEST.md",
                    "examples/one_sun.crf.json"):
                    self.assertIn("coherent-diffusion/"+path,names)
                    self.assertTrue(archive.read("coherent-diffusion/"+path))
                self.assertFalse(any("/tests/" in x or "__pycache__" in x or x.endswith(".pyc")
                                     for x in names))

    def test_missing_entrypoint_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                builder.members(temporary)


if __name__ == "__main__":
    unittest.main()
