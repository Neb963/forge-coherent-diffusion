"""Deterministic, source-only Forge Neo extension archive (P07 candidate).

No bundled model weights, vendored Forge internals, network assets or caches.
The CI ZIP is a candidate until installed-model evidence passes P07 gates.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "coherent-diffusion/"
STATIC = ("README.md", "requirements.txt", "INSTALL_AND_TEST.md", "examples/one_sun.crf.json")


def members(root=ROOT):
    base = Path(root)
    paths = [base / rel for rel in STATIC]
    for sub, extensions in (("coherent_diffusion", {".py", ".json", ".html"}),
                            ("scripts", {".py"})):
        paths.extend(p for p in (base / sub).rglob("*")
                     if p.is_file() and p.suffix in extensions
                     and "__pycache__" not in p.parts)
    for file in paths:
        if not file.is_file() or file.is_symlink() or not file.resolve().is_relative_to(base.resolve()):
            raise ValueError("Missing/unsafe release member: " + str(file))
    names = [p.relative_to(base).as_posix() for p in paths]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate release member")
    if "scripts/coherent_diffusion.py" not in names or "coherent_diffusion/editor/index.html" not in names:
        raise ValueError("Missing installed Forge extension entrypoint/frontend")
    return sorted(zip(names, paths))


def build(output, root=ROOT):
    target=Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for rel, source in members(root):
            info=ZipInfo(PREFIX+rel, (2026, 10, 25, 0, 0, 0))
            info.compress_type=ZIP_DEFLATED
            info.external_attr=0o100644 << 16
            archive.writestr(info, source.read_bytes(), compress_type=ZIP_DEFLATED,
                             compresslevel=9)
    return target


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output", default="dist/coherent-diffusion-P07-candidate.zip")
    args=p.parse_args()
    print(build(args.output))
