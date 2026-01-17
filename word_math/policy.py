from .config import load_config
from .features import compute_y, compute_z
from .scoring import compute_score, ScoreResult
import json
from datetime import datetime
import numpy as np

class WordMathGuard:
  def __init__(self, config_path: str):
    self.cfg = load_config(config_path)

  def assess(
      self,
      tokens: list[str],
      msg_embedding: np.ndarray,
      topic_embedding: np.ndarray,
  ) -> ScoreResult:
    y = compute_y(tokens)
    z = compute_z(msg_embedding, topic_embedding)
    result = compute_score(
      y, z,
      self.cfg.scoring,
      self.cfg.thresholds,
    )
    if self.cfg.logging.enable_json_logs and self.cfg.experiment.save_scores:
      self._log(result)
    return result

  def _log(self, result: ScoreResult) -> None:
    record = {
      "timestamp": datetime.utcnow().isoformat(),
      "y": result.y,
      "z": result.z,
      "f": result.f,
      "risk_band": result.risk_band,
      "triggers": result.triggers,
      "hex": f"{self.cfg.logging.hex_namespace}[{result.hex_trace}]",
    }
    with open(self.cfg.experiment.output_path, "a") as f:
      f.write(json.dumps(record) + "\n")

  def should_block(self, result: ScoreResult) -> bool:
    return result.risk_band == "high"

  def should_rewrite(self, result: ScoreResult) -> bool:
    return (
      result.risk_band == "medium"
      or "REPEAT_TRIGGER" in result.triggers
      or "DRIFT_TRIGGER" in result.triggers
    )
