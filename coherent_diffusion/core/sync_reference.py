"""CPU-only synthetic synchronized tile-prediction reference (NOT a GPU sampler)."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True, order=True)
class Window:
    x0: int
    y0: int
    x1: int
    y1: int

def _starts(length: int, size: int, overlap: int) -> list[int]:
    if size>=length: return [0]
    starts=list(range(0,length-size+1,size-overlap))
    if starts[-1]!=length-size: starts.append(length-size)
    return starts

def plan_windows(width: int, height: int, tile: int, overlap: int) -> list[Window]:
    if min(width,height,tile)<=0 or overlap<0 or overlap>=tile:
        raise ValueError("Positive canvas/tile and 0 <= overlap < tile required")
    return [Window(x,y,min(width,x+tile),min(height,y+tile))
            for y in _starts(height,tile,overlap)
            for x in _starts(width,tile,overlap)]

def synchronized_step(
    latent: list[list[float]],
    windows: list[Window],
    predict: Callable[[tuple[tuple[float,...],...],Window],list[list[float]]],
    advance: Callable[[float,float],float],
) -> list[list[float]]:
    """Freeze entire step, collect weighted predictions, advance once per pixel.

    Each predictor receives a *tuple* crop from the unchanged global state.
    Order-independent accumulation via math.fsum(sorted(...)); positive cosine-ish
    edge taper that never zeroes any core pixel, including the outer boundary.
    """
    h=len(latent)
    if not h or not latent[0] or any(len(row)!=len(latent[0]) for row in latent):
        raise ValueError("Nonempty rectangular global latent required")
    w=len(latent[0])
    frozen=tuple(tuple(float(v) for v in row) for row in latent)
    numerator=[[[] for _ in range(w)] for _ in range(h)]
    denominator=[[[] for _ in range(w)] for _ in range(h)]
    for win in windows:
        if not 0<=win.x0<win.x1<=w or not 0<=win.y0<win.y1<=h:
            raise ValueError("Tile outside full latent canvas")
        crop=tuple(row[win.x0:win.x1] for row in frozen[win.y0:win.y1])
        predicted=predict(crop,win)
        if len(predicted)!=win.y1-win.y0 or any(len(r)!=win.x1-win.x0 for r in predicted):
            raise ValueError("Predictor output geometry mismatch")
        for iy,row in enumerate(predicted):
            for ix,v in enumerate(row):
                if not math.isfinite(v):
                    raise ValueError("Non-finite predictor output")
                # Positive interior weighting, not independent crop noise.
                dx=min(ix+1,win.x1-win.x0-ix)
                dy=min(iy+1,win.y1-win.y0-iy)
                weight=float(dx*dy)
                x,y=win.x0+ix,win.y0+iy
                numerator[y][x].append(weight*float(v))
                denominator[y][x].append(weight)
    out=[]
    for y in range(h):
        row=[]
        for x in range(w):
            weight=math.fsum(sorted(denominator[y][x]))
            if weight<=0: raise ValueError(f"Uncovered global pixel at ({x},{y})")
            p=math.fsum(sorted(numerator[y][x]))/weight
            row.append(float(advance(frozen[y][x],p)))
        out.append(row)
    return out
