"""Secure, bounded, data-only CRF archive reader. Never extracts files to disk."""
from __future__ import annotations

import hashlib
import io
import json
import struct
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

from .validation import CRFError, ValidationReport, parse_validate, _safe_asset

MAX_ARCHIVE_BYTES = 24 * 1024 * 1024
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_INFLATED_BYTES = 64 * 1024 * 1024
MAX_MEMBERS = 132


@dataclass(frozen=True)
class ImportedArchive:
    report: ValidationReport
    assets: dict[str, bytes]


def _image_size(data: bytes, media_type: str) -> tuple[int, int]:
    """Parse image headers without letting Pillow or external codecs decode input."""
    if media_type == "image/png":
        if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
            raise CRFError("Mask is not a PNG")
        length = struct.unpack_from(">I", data, 8)[0]
        if length != 13:
            raise CRFError("Invalid PNG IHDR")
        w, h = struct.unpack_from(">II", data, 16)
        return w, h
    if media_type == "image/webp":
        if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
            raise CRFError("Mask is not WebP")
        fourcc = data[12:16]
        if fourcc == b"VP8X":
            return (1 + int.from_bytes(data[24:27], "little"),
                    1 + int.from_bytes(data[27:30], "little"))
        if fourcc == b"VP8 " and len(data) >= 30 and data[23:26] == b"\x9d\x01\x2a":
            return (int.from_bytes(data[26:28], "little") & 0x3fff,
                    int.from_bytes(data[28:30], "little") & 0x3fff)
        if fourcc == b"VP8L" and len(data) >= 25 and data[20] == 0x2f:
            bits = int.from_bytes(data[21:25], "little")
            return ((bits & 0x3fff) + 1, ((bits >> 14) & 0x3fff) + 1)
        raise CRFError("Unsupported WebP mask header")
    raise CRFError("Unsupported mask media_type")


def load_crf_zip(raw: bytes) -> ImportedArchive:
    """Validate CRF zip, SHA256, dimensions, filenames, and decompression budget.

    The returned asset bytes are opaque; P06 must still safely decode pixels
    and build generation masks. No side effects, network or filesystem writes.
    """
    if not isinstance(raw, bytes) or len(raw) > MAX_ARCHIVE_BYTES:
        raise CRFError("CRF archive exceeds 24 MiB")
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except (OSError, zipfile.BadZipFile) as exc:
        raise CRFError("Invalid ZIP archive") from exc
    with zf:
        members = zf.infolist()
        if len(members) > MAX_MEMBERS:
            raise CRFError("Too many ZIP entries")
        seen = set()
        inflated = 0
        for m in members:
            path = m.filename
            if path in seen:
                raise CRFError(f"Duplicate archive path: {path}")
            seen.add(path)
            if m.is_dir() or not _safe_asset(path) or path not in ("composition.json", "manifest.json") and not path.startswith("masks/"):
                raise CRFError(f"Unsafe/unexpected archive path: {path}")
            if m.flag_bits & 1 or m.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise CRFError(f"Unsupported/encrypted entry: {path}")
            if (m.external_attr >> 16) & 0o170000 == 0o120000:
                raise CRFError(f"Symlink entry forbidden: {path}")
            if m.file_size > MAX_MEMBER_BYTES or m.file_size > max(1, m.compress_size) * 200:
                raise CRFError(f"Entry size/ratio exceeds safety limit: {path}")
            inflated += m.file_size
        if inflated > MAX_INFLATED_BYTES or "composition.json" not in seen:
            raise CRFError("Missing composition.json or archive too large")
        try:
            json_raw = zf.read("composition.json")
            report = parse_validate(json_raw.decode("utf-8"))
        except (UnicodeError, zipfile.BadZipFile, RuntimeError, EOFError, OSError) as exc:
            raise CRFError("Cannot read composition.json") from exc
        declared = {x["path"]: x for x in report.document.get("assets", [])}
        if set(declared) != seen - {"composition.json", "manifest.json"}:
            raise CRFError("ZIP members do not match declared assets")
        if "manifest.json" in seen:
            try:
                manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            except (ValueError, UnicodeError) as exc:
                raise CRFError("Invalid manifest.json") from exc
            if manifest != {"format": "coherent-regions-archive", "version": 1}:
                raise CRFError("Unsupported manifest")
        result = {}
        for path, meta in declared.items():
            try:
                data = zf.read(path)
            except (zipfile.BadZipFile, RuntimeError, EOFError, OSError) as exc:
                raise CRFError(f"Damaged ZIP member: {path}") from exc
            if hashlib.sha256(data).hexdigest() != meta["sha256"]:
                raise CRFError(f"Mask checksum mismatch: {path}")
            size = _image_size(data, meta["media_type"])
            if size != (meta["width"], meta["height"]) or size[0] * size[1] > 16_777_216:
                raise CRFError(f"Mask dimensions mismatch/limit: {path}")
            result[meta["id"]] = data
        return ImportedArchive(report, result)
