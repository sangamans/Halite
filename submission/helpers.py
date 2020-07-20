from typing import Optional
import numpy as np

# helpers for MCTS

MAXIMUM_FLOAT_VALUE = float('inf')
# holds min max of the tree
class MinMaxStats(object):
  def __init__(self, known_bounds: Optional[KnownBounds]):
    self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
    self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

  def update(self, value: float):
    self.maximum = max(self.maximum, value)
    self.minimum = min(self.minimum, value)

  def normalize(self, value: float) -> float:
    if self.maximum > self.minimum:
      # normalize only when set the maximum and minimum values.
      return (value - self.minimum) / (self.maximum - self.minimum)
    return value

# nodes inside mcts
class Node(object):

  def __init__(self, prior: float):
    self.visit_count = 0
    self.to_play = -1
    self.prior = prior
    self.value_sum = 0
    self.children = {}
    self.hidden_state = None
    self.reward = 0

  def expanded(self) -> bool:
    return len(self.children) > 0

  def value(self) -> Optional[float]:
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count

def softmax_sample(visit_counts, actions, t):
    counts_exp = np.exp(visit_counts) * (1 / t)
    probs = counts_exp / np.sum(counts_exp, axis=0)
    action_idx = np.random.choice(len(actions), p=probs)
    return actions[action_idx]