"""Opt-in synchronized SDXL Euler and Anima one-frame flow executors (P03–P05)."""
from .sdxl_sync import TilePlan, Tile, Rect, RunResult, synchronized_euler, seeded_noise
from .anima_flow import (
    AnimaQualityPlan, AnimaQualityResult, AnimaResult,
    anima_noise, execute_anima, flow_sigmas, synchronized_flow,
    validate_anima_tiles,
)
__all__ = [
    "TilePlan", "Tile", "Rect", "RunResult", "synchronized_euler", "seeded_noise",
    "AnimaQualityPlan", "AnimaQualityResult", "AnimaResult",
    "anima_noise", "execute_anima", "flow_sigmas", "synchronized_flow",
    "validate_anima_tiles",
]
