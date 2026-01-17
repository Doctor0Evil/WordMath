from collections import Counter
from typing import Sequence
import numpy as np

def compute_y(tokens: Sequence[str]) -> float:
  if not tokens:
    return 0.0
  counts = Counter(tokens)
  n = len(tokens)
  max_freq = max(counts.values()) / n
  return float(max_freq)

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
  if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
    return 1.0
  sim = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
  return 0.5 * (1.0 - sim)  # map [-1,1] -> [0,1]

def compute_z(
    msg_embedding: np.ndarray,
    topic_embedding: np.ndarray,
) -> float:
  return cosine_distance(msg_embedding, topic_embedding)
