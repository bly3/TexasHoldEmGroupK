"""Utility functions to save/load strategy tables."""

import pickle
from typing import Dict


def save_strategy(path: str, strategy: Dict[str, Dict[str, float]]) -> None:
    with open(path, "wb") as f:
        pickle.dump(strategy, f)


def load_strategy(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, "rb") as f:
        return pickle.load(f)

