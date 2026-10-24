"""Data-only CRF v1 parsing and semantic checks. No Forge imports."""
from __future__ import annotations
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

MAX_BYTES = 1_048_576
TOL = 1e-9
MIN_AREA = 1e-6
FENCE = re.compile(r"\A\s*\x60{3}(?:json|crf)\s*\n(.*?)\n\x60{3}\s*\Z", re.I | re.S)

class CRFError(ValueError):
    pass

@dataclass(frozen=True)
class ValidationReport:
    document: dict[str, Any]
    warnings: tuple[str, ...]

def _pairs(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise CRFError(f"Duplicate JSON key: {k}")
        out[k] = v
    return out

def _nonfinite(value):
    raise CRFError(f"Non-finite JSON literal: {value}")

def _depth(value, n=0):
    if n > 32:
        raise CRFError("JSON depth limit exceeded")
    if isinstance(value, dict):
        for v in value.values(): _depth(v, n + 1)
    elif isinstance(value, list):
        for v in value: _depth(v, n + 1)

def parse_crf(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_BYTES:
        raise CRFError("CRF must be UTF-8 text under 1 MiB")
    source = text.strip()
    if source.startswith(chr(96) * 3):
        m = FENCE.fullmatch(source)
        if not m:
            raise CRFError("Exactly one fenced json/crf block required")
        source = m.group(1)
    try:
        result = json.loads(source, object_pairs_hook=_pairs, parse_constant=_nonfinite)
    except (ValueError, TypeError) as exc:
        raise CRFError(f"Invalid JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise CRFError("CRF root must be a JSON object")
    _depth(result)
    return result

def canonical_json(doc: dict[str, Any]) -> str:
    return json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

def _schema_check(doc):
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise CRFError("Install jsonschema from requirements.txt") from exc
    schema = json.loads(Path(__file__).with_name("schema_v1.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(doc),
                    key=lambda e: (str(list(e.absolute_path)), e.message))
    if errors:
        e = errors[0]
        raise CRFError(f"Schema at {list(e.absolute_path)}: {e.message}")

def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def _on(a, b, c):
    return abs(_cross(a,b,c)) <= TOL and min(a[0],b[0])-TOL <= c[0] <= max(a[0],b[0])+TOL and min(a[1],b[1])-TOL <= c[1] <= max(a[1],b[1])+TOL

def _intersects(a,b,c,d):
    u,v,w,z = _cross(a,b,c),_cross(a,b,d),_cross(c,d,a),_cross(c,d,b)
    if ((u>TOL and v < -TOL) or (u < -TOL and v>TOL)) and ((w>TOL and z < -TOL) or (w < -TOL and z>TOL)):
        return True
    return _on(a,b,c) or _on(a,b,d) or _on(c,d,a) or _on(c,d,b)

def _polygon(points, rid):
    n = len(points)
    area = abs(sum(points[i][0]*points[(i+1)%n][1] - points[(i+1)%n][0]*points[i][1] for i in range(n))/2)
    if area < MIN_AREA: raise CRFError(f"{rid}: polygon area too small")
    for i in range(n):
        a,b = points[i],points[(i+1)%n]
        if math.dist(a,b) < TOL: raise CRFError(f"{rid}: zero-length edge")
        for j in range(i+1,n):
            if j==(i+1)%n or i==(j+1)%n: continue
            if _intersects(a,b,points[j],points[(j+1)%n]):
                raise CRFError(f"{rid}: self-intersecting polygon")

def _safe_asset(path):
    return isinstance(path,str) and len(path) <= 256 and bool(path) and not path.startswith("/") and "\\" not in path and ":" not in path and all(p not in ("",".","..") for p in path.split("/")) and str(PurePosixPath(path))==path and len(path.split("/"))<=8

def validate_crf(doc: dict[str, Any], *, target_canvas: tuple[int,int] | None = None) -> ValidationReport:
    """Validate without mutating state, touching external assets, or invoking a model."""
    _schema_check(doc)
    warnings = []
    ids, lora_ids = set(), set()
    for region in doc["regions"]:
        rid = region["id"]
        if rid in ids: raise CRFError(f"Duplicate region ID: {rid}")
        ids.add(rid)
        g=region["geometry"]
        if g["type"]=="bbox":
            x,y,w,h=g["bbox"]
            if w <= 0 or h <= 0 or w*h < MIN_AREA: raise CRFError(f"{rid}: bbox area too small")
            if x+w > 1+TOL or y+h > 1+TOL: raise CRFError(f"{rid}: bbox outside canvas")
        elif g["type"]=="polygon": _polygon(g["points"],rid)
        else: warnings.append(f"{rid}: binary mask/hash not yet verified; P02 ZIP importer required")
        for a in region.get("loras",[]):
            if a["id"] in lora_ids: raise CRFError(f"Duplicate adapter ID: {a['id']}")
            lora_ids.add(a["id"])
            if ".." in a["name"]: raise CRFError(f"{rid}: unsafe symbolic adapter name")
            if a.get("te_weight") is not None:
                warnings.append(f"{rid}: text-encoder LoRA isolation unverified until P06")
        if region["blend"]["feather_px"] > min(doc["canvas"]["width"],doc["canvas"]["height"]):
            warnings.append(f"{rid}: feather exceeds canvas extent, clamp on compile")
    entity_ids=set()
    for e in doc.get("entities",[]):
        if e["id"] in entity_ids: raise CRFError(f"Duplicate entity ID: {e['id']}")
        entity_ids.add(e["id"])
        for rid in e["region_ids"]:
            if rid not in ids: raise CRFError(f"Entity {e['id']} references missing region {rid}")
    assets={}
    paths=set()
    for a in doc.get("assets",[]):
        if a["id"] in assets or a["path"] in paths: raise CRFError("Duplicate asset id/path")
        if not _safe_asset(a["path"]): raise CRFError(f"Unsafe asset path: {a['path']}")
        assets[a["id"]]=a
        paths.add(a["path"])
    used={r["geometry"]["asset_id"] for r in doc["regions"] if r["geometry"]["type"]=="mask"}
    if missing:=used-set(assets): raise CRFError(f"Missing mask assets: {sorted(missing)}")
    if orphan:=set(assets)-used: raise CRFError(f"Orphan assets: {sorted(orphan)}")
    if doc["global"]["base_weight"] == 0:
        def full(r):
            g=r["geometry"]
            return g["type"]=="bbox" and all(abs(a-b)<=TOL for a,b in zip(g["bbox"],(0,0,1,1)))
        if not any(r.get("enabled",True) and r["blend"]["weight"]>0 and full(r) for r in doc["regions"]):
            raise CRFError("Zero global base weight requires enabled positive full-canvas bbox (conservative exact coverage)")
    if target_canvas is not None:
        w,h=target_canvas
        if w<=0 or h<=0: raise CRFError("Invalid target canvas")
        a=doc["canvas"]["width"]/doc["canvas"]["height"]
        b=w/h
        if not math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9):
            warnings.append(f"Aspect mismatch {a:.5g} vs {b:.5g}: preview {doc['canvas']['aspect_policy']} mapping before apply")
    if doc["blend"]["policy"]=="priority":
        warnings.append("Priority tie-breaker: z_index, then original document order")
    return ValidationReport(doc,tuple(warnings))

def parse_validate(text: str, **kwargs) -> ValidationReport:
    return validate_crf(parse_crf(text),**kwargs)
