"""
Lightweight opponent model tracking simple tendencies.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List


class OpponentModel:
    def __init__(self, window: int = 100):
        self.window = window
        self.actions: Deque[str] = deque(maxlen=window)
        self.fold_vs_bet: int = 0
        self.faced_bet: int = 0

    def update(self, outcome: Dict) -> None:
        """
        outcome can include:
        - 'opponent_actions': list of their actions this hand
        - 'faced_bet' (bool)
        - 'folded_vs_bet' (bool)
        """
        for a in outcome.get("opponent_actions", []):
            self.actions.append(a)
        if outcome.get("faced_bet"):
            self.faced_bet += 1
            if outcome.get("folded_vs_bet"):
                self.fold_vs_bet += 1

    def is_overfolding(self) -> bool:
        if self.faced_bet < 5:
            return False
        fold_rate = self.fold_vs_bet / max(1, self.faced_bet)
        return fold_rate > 0.55

    def is_calling_station(self) -> bool:
        if self.faced_bet < 5:
            return False
        fold_rate = self.fold_vs_bet / max(1, self.faced_bet)
        return fold_rate < 0.25

