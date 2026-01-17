from dataclasses import dataclass
from typing import Literal
import yaml

ScoreVariant = Literal["linear", "quadratic", "interaction"]

@dataclass
class ScoringConfig:
  variant: ScoreVariant
  alpha: float
  beta: float
  gamma: float | None = None

@dataclass
class ThresholdConfig:
  high_risk_max: float
  medium_risk_max: float
  max_repetition: float
  max_drift: float

@dataclass
class LoggingConfig:
  hex_namespace: str
  enable_json_logs: bool

@dataclass
class ExperimentConfig:
  save_scores: bool
  output_path: str

@dataclass
class WordMathConfig:
  scoring: ScoringConfig
  thresholds: ThresholdConfig
  logging: LoggingConfig
  experiment: ExperimentConfig

def load_config(path: str) -> WordMathConfig:
  with open(path, "r") as f:
    raw = yaml.safe_load(f)
  s = raw["scoring"]; t = raw["thresholds"]
  l = raw["logging"]; e = raw["experiment"]
  return WordMathConfig(
    scoring=ScoringConfig(**s),
    thresholds=ThresholdConfig(**t),
    logging=LoggingConfig(**l),
    experiment=ExperimentConfig(**e),
  )
