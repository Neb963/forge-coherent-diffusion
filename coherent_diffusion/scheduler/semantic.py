"""P04 patch-aware semantic routing plan, not a hooked text-conditioning backend.

The global scene is preserved for a global anchor; local windows inherit style
and only applicable authored region/entity content. Geometry is CRF-normalized;
entity IDs/counts never split or multiply when crossing tile boundaries.
Automatic proposals are explicitly UNLOCATED and not silently assigned to tiles.
Actual model attention/LoRA/control routing is implemented in P06.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from coherent_diffusion.crf.validation import validate_crf
from coherent_diffusion.scheduler.sdxl_sync import TilePlan

_TOL = 1e-12
_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
_COUNTS = re.compile(
    r"\b(?:exactly\s+)?(one|two|three|four|five|[1-9][0-9]?)\s+"
    r"([A-Za-z][A-Za-z-]{2,})\b", re.IGNORECASE
)


@dataclass(frozen=True)
class EntityPlacement:
    id: str
    description: str
    count: int
    region_ids: tuple[str, ...]
    # Union of authored region bounding boxes, not an inferred object silhouette.
    normalized_bounds: tuple[float, float, float, float]


@dataclass(frozen=True)
class UnlocatedProposal:
    phrase: str
    count: int
    confidence: float
    # Explicitly cannot drive a tile mask without further user/anchor evidence.
    located: bool = False


@dataclass(frozen=True)
class PatchRoute:
    tile_index: int
    normalized_read: tuple[float, float, float, float]
    region_ids: tuple[str, ...]
    entity_ids: tuple[str, ...]
    local_prompt: str
    local_negative_prompts: tuple[str, ...]


@dataclass(frozen=True)
class RoutingPlan:
    mode: str
    global_scene_prompt: str
    global_style_prompt: str
    global_negative_prompt: str | None
    anchor_prompt: str
    entities: tuple[EntityPlacement, ...]
    proposals: tuple[UnlocatedProposal, ...]
    patches: tuple[PatchRoute, ...]
    warnings: tuple[str, ...]
    connected_to_model: bool = False


def _bounds(geom):
    if geom["type"] == "bbox":
        x, y, w, h = geom["bbox"]
        return (x, y, x+w, y+h)
    if geom["type"] == "polygon":
        pts = geom["points"]
        return (min(p[0] for p in pts), min(p[1] for p in pts),
                max(p[0] for p in pts), max(p[1] for p in pts))
    raise ValueError("P04 requires explicit bbox/polygon spatial geometry; masks route in P06")


def _clip_polygon(points, rect):
    """Convex rectangle clipping for arbitrary simple authored polygons."""
    x0, y0, x1, y1 = rect
    poly = [tuple(p) for p in points]
    # Boundary inclusive; final area test prevents zero-overlap adjacency.
    planes = (
        (lambda p: p[0]-x0, lambda a, b, t: (x0, a[1]+t*(b[1]-a[1]))),
        (lambda p: x1-p[0], lambda a, b, t: (x1, a[1]+t*(b[1]-a[1]))),
        (lambda p: p[1]-y0, lambda a, b, t: (a[0]+t*(b[0]-a[0]), y0)),
        (lambda p: y1-p[1], lambda a, b, t: (a[0]+t*(b[0]-a[0]), y1)),
    )
    for signed, intersection in planes:
        out = []
        if not poly:
            break
        previous = poly[-1]
        prev_d = signed(previous)
        for point in poly:
            curr_d = signed(point)
            prev_in, curr_in = prev_d >= -_TOL, curr_d >= -_TOL
            if prev_in != curr_in:
                t = prev_d / (prev_d - curr_d)
                out.append(intersection(previous, point, t))
            if curr_in:
                out.append(point)
            previous, prev_d = point, curr_d
        poly = out
    if len(poly) < 3:
        return False
    area = abs(sum(poly[i][0] * poly[(i+1) % len(poly)][1]
                   - poly[(i+1) % len(poly)][0] * poly[i][1]
                   for i in range(len(poly))) / 2)
    return area > _TOL


def _intersects(geometry, rect):
    x0, y0, x1, y1 = _bounds(geometry)
    if min(x1, rect[2])-max(x0,rect[0]) <= _TOL or min(y1,rect[3])-max(y0,rect[1]) <= _TOL:
        return False
    if geometry["type"] == "bbox":
        return True
    return _clip_polygon(geometry["points"], rect)


def _unique_join(parts):
    seen, chunks = set(), []
    for part in parts:
        if not isinstance(part, str):
            continue
        part = part.strip()
        if part and part not in seen:
            seen.add(part)
            chunks.append(part)
    return ", ".join(chunks)


def _automatic_proposals(scene):
    found = {}
    for match in _COUNTS.finditer(scene):
        number = match.group(1).lower()
        count = _NUMBERS.get(number, int(number) if number.isdecimal() else 0)
        noun = match.group(2).lower()
        if noun not in found:
            found[noun] = UnlocatedProposal(match.group(0), count, .25)
    return tuple(found.values())


def compile_routes(doc: dict, plan: TilePlan, *, mode: str = "manual") -> RoutingPlan:
    """Compile deterministic non-mutating *advisory* prompt routing for P04.

    manual: explicit regions and entities;
    automatic: route only unambiguous noun-to-authored-region matches;
    hybrid: authored regions plus such conservative automatic matches.
    Unmatched proposals stay explicitly unlocated; never fake a position.
    """
    if mode not in ("manual", "automatic", "hybrid"):
        raise ValueError("Routing mode must be manual, automatic or hybrid")
    validate_crf(doc, target_canvas=(plan.width_px,plan.height_px))
    if doc["canvas"]["width"] * plan.height_px != doc["canvas"]["height"] * plan.width_px:
        raise ValueError("P04 requires matching scene/target aspect; explicit reproject belongs to P06")
    scene = doc["global"]["scene_prompt"]
    style = doc["global"]["style_prompt"]
    neg = doc["global"].get("negative_prompt")
    warnings = ["P04 routes are advisory; SDXL/Anima text conditioning is not connected until P06"]
    enabled = [r for r in doc["regions"] if r.get("enabled", True) and r["blend"]["weight"] > 0]
    # Mask manifests require verified binary pixels. Do not silently approximate
    # masks using their bounding box or apply regional content globally.
    if mode != "automatic" and any(r["geometry"]["type"] == "mask" for r in enabled):
        raise ValueError("P04 cannot route binary mask assets without P06 mask compiler")
    regions = {r["id"]: r for r in enabled if r["geometry"]["type"] != "mask"}
    entities = []
    for entity in doc.get("entities", []):
        ids = tuple(rid for rid in entity["region_ids"] if rid in regions)
        if not ids:
            warnings.append("Entity " + entity["id"] + " has no active spatial regions")
            continue
        boxes = [_bounds(regions[rid]["geometry"]) for rid in ids]
        bounds = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                  max(b[2] for b in boxes), max(b[3] for b in boxes))
        entities.append(EntityPlacement(entity["id"], entity["description"],
                                        entity["count"], ids, bounds))
    matched_regions = set()
    unresolved = []
    for proposal in (_automatic_proposals(scene) if mode in ("automatic", "hybrid") else ()):
        noun = proposal.phrase.split()[-1].lower()
        candidates = [r for r in enabled if r["id"].lower() == noun
                      or r.get("name", "").lower() == noun]
        if len(candidates) != 1:
            unresolved.append(proposal)
            continue
        found = candidates[0]
        rid = found["id"]
        if found["geometry"]["type"] == "mask":
            warnings.append("Automatic match " + noun + " requires a verified P06 mask asset")
            unresolved.append(proposal)
            continue
        matched_regions.add(rid)
        existing = [e for e in entities if rid in e.region_ids]
        if existing:
            if any(e.count != proposal.count for e in existing):
                warnings.append("Count proposal conflicts with authored entity at " + rid
                                + "; authored count retained")
        else:
            new_id = "auto." + noun
            if any(e.id == new_id for e in entities):
                unresolved.append(proposal)
                continue
            r_bounds = _bounds(found["geometry"])
            entities.append(EntityPlacement(new_id, proposal.phrase,
                                            proposal.count, (rid,), r_bounds))
        warnings.append("Automatic match " + noun + " uses pre-authored region "
                        + rid + ", not inferred placement")
    if mode == "automatic":
        entities = [e for e in entities
                    if any(rid in matched_regions for rid in e.region_ids)]
    proposals = tuple(unresolved)
    if proposals:
        warnings.append("Unlocated count suggestions require operator/anchor placement before spatial use")
    patches = []
    for tile in plan.windows:
        r = tile.read
        rect = (r.x0/plan.width, r.y0/plan.height,
                r.x1/plan.width, r.y1/plan.height)
        ids = tuple(rid for rid, region in regions.items()
                    if (mode != "automatic" or rid in matched_regions)
                    and _intersects(region["geometry"], rect))
        patch_entities = tuple(e.id for e in entities
                               if any(rid in ids for rid in e.region_ids))
        applicable = [regions[rid] for rid in ids]
        local_parts = [style] if not applicable else []
        for region in applicable:
            parts = []
            if region.get("inherit_style", True):
                parts.append(style)
            if region.get("prompt") is not None:
                parts.append(region["prompt"])
            elif any(e.id in patch_entities and region["id"] in e.region_ids for e in entities):
                parts.extend(e.description for e in entities if e.id in patch_entities and region["id"] in e.region_ids)
            local_parts.extend(parts)
        # Never re-broadcast full global scene to every local tile.
        patches.append(PatchRoute(tile.index, rect, ids, patch_entities,
                                  _unique_join(local_parts),
                                  tuple(region["negative_prompt"] for region in applicable
                                        if region.get("negative_prompt") is not None)))
    return RoutingPlan(mode, scene, style, neg, _unique_join((scene,style)),
                       tuple(entities), proposals, tuple(patches),
                       tuple(warnings))
