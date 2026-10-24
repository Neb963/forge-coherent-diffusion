"""Deterministic CPU vector geometry rasterization; no model/cuda dependencies."""
from __future__ import annotations
from .validation import CRFError, _on

MAX_RASTER_PIXELS = 1_048_576

def _inside_polygon(x: float, y: float, points: list[list[float]]) -> bool:
    # Even/odd ray casting, with polygon boundary included. Evaluate pixel centers.
    inside = False
    for i in range(len(points)):
        a, b = points[i], points[(i+1) % len(points)]
        if _on(a,b,[x,y]):
            return True
        if (a[1] > y) != (b[1] > y):
            intersection = (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]
            if x < intersection:
                inside = not inside
    return inside

def rasterize(geometry: dict, width: int, height: int) -> list[list[float]]:
    """Binary mask at pixel centers ((x+.5)/width,(y+.5)/height), [row][column].

    Bbox upper edges are exclusive; polygons use even/odd fill with boundaries
    included. No implicit feathering, viewport transform or regeneration tile offset.
    """
    if type(width) is not int or type(height) is not int or width<=0 or height<=0:
        raise CRFError("Raster size must contain positive integer dimensions")
    if width*height > MAX_RASTER_PIXELS:
        raise CRFError("CPU raster exceeds 1M pixels; choose a smaller diagnostic grid")
    typ=geometry.get("type")
    if typ=="bbox":
        x,y,w,h=geometry["bbox"]
        if w<=0 or h<=0 or x<0 or y<0 or x+w>1+1e-9 or y+h>1+1e-9:
            raise CRFError("Invalid bbox; validate CRF first")
        def inside(px,py):
            return x <= px < x+w and y <= py < y+h
    elif typ=="polygon":
        points=geometry["points"]
        # Caller should run validate_crf first for self-intersection.
        def inside(px,py):
            return _inside_polygon(px,py,points)
    elif typ=="mask":
        raise CRFError("Mask references require verified binary asset importer in P02")
    else:
        raise CRFError("Unknown geometry type")
    return [[1.0 if inside((xx+.5)/width,(yy+.5)/height) else 0.0
             for xx in range(width)] for yy in range(height)]
