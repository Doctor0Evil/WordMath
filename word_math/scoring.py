from __future__ import annotations
from dataclasses import dataclass
from .config import ScoringConfig, ThresholdConfig
import math
import uuid
import json
from datetime import datetime

@dataclass
class ScoreResult:
  y: float
  z: float
  f: float
  risk_band: str          # "high" | "medium" | "low"
  hex_trace: str          # hex-stamped id for logs
  triggers: list[str]     # e.g., ["REPEAT_TRIGGER", "DRIFT_TRIGGER"]

def _clip01(x: float) -> float:
  return max(0.0, min(1.0, x))

def score_linear(y: float, z: float, cfg: ScoringConfig) -> float:
  f = 1.0 - cfg.alpha * y - cfg.beta * z
  return _clip01(f)

def score_quadratic(y: float, z: float, cfg: ScoringConfig) -> float:
  f = 1.0 - cfg.alpha * (y ** 2) - cfg.beta * (z ** 2)
  return _clip01(f)

def score_interaction(y: float, z: float, cfg: ScoringConfig) -> float:
  gamma = cfg.gamma or 0.0
  f = 1.0 - cfg.alpha * y - cfg.beta * z - gamma * y * z
  return _clip01(f)

def compute_score(
    y: float,
    z: float,
    s_cfg: ScoringConfig,
    t_cfg: ThresholdConfig,
) -> ScoreResult:
  y = _clip01(y); z = _clip01(z)

  if s_cfg.variant == "linear":
    f = score_linear(y, z, s_cfg)
  elif s_cfg.variant == "quadratic":
    f = score_quadratic(y, z, s_cfg)
  elif s_cfg.variant == "interaction":
    f = score_interaction(y, z, s_cfg)
  else:
    raise ValueError(f"Unknown variant: {s_cfg.variant}")

  if f <= t_cfg.high_risk_max:
    band = "high"
  elif f <= t_cfg.medium_risk_max:
    band = "medium"
  else:
    band = "low"

  triggers: list[str] = []
  if y > t_cfg.max_repetition:
    triggers.append("REPEAT_TRIGGER")
  if z > t_cfg.max_drift:
    triggers.append("DRIFT_TRIGGER")

  hex_trace = uuid.uuid4().hex  # e.g. "2fa9e3..."

  return ScoreResult(
    y=y, z=z, f=f,
    risk_band=band,
    hex_trace=hex_trace,
    triggers=triggers,
  )
