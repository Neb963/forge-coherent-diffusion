"""Honest bounded benchmark metadata; measurements supplied only by caller."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

@dataclass
class BenchmarkRecord:
    model_family: str
    mode: str
    forge_commit: str | None = None
    encoder_route: str | None = None
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    status: str = "not_tested"
    measurements: dict[str,float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str,Any]:
        if self.status not in ("not_tested","pass","fail","blocked"):
            raise ValueError("Invalid benchmark status")
        if len(self.notes)>40 or len(self.measurements)>32:
            raise ValueError("Benchmark metadata limit exceeded")
        if self.status == "not_tested" and self.measurements:
            raise ValueError("Do not attach measurements to not_tested records")
        import math
        for k,v in self.measurements.items():
            if not isinstance(v,(float,int)) or not math.isfinite(v):
                raise ValueError(f"Invalid numeric metric: {k}")
        return {"timestamp_utc":datetime.now(timezone.utc).isoformat(),
                "model_family":self.model_family,"mode":self.mode,
                "forge_commit":self.forge_commit,"encoder_route":self.encoder_route,
                "width":self.width,"height":self.height,"seed":self.seed,
                "status":self.status,"measurements":self.measurements,"notes":self.notes}

    def save(self,path: str|Path) -> None:
        """One JSON record per line; caller selects destination, no remote upload."""
        record=self.to_dict()
        p=Path(path)
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open("a",encoding="utf-8") as f:
            f.write(json.dumps(record,sort_keys=True,allow_nan=False)+"\n")
