from __future__ import annotations

import math
from typing import Dict, Optional

from poker_cfr.cfr.trainer import CFRTrainer
from poker_cfr.env.game import GameState


class RealtimeResolver:
    """
    Lightweight real-time subgame resolver.

    Instantiate once and call resolve() with a GameState that represents the
    current environment state. The resolver runs a short CFR/MCCFR search from
    that node and returns action probabilities for the root player.
    """

    def __init__(self, iterations: int = 800, algo: str = "mccfr") -> None:
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if algo not in {"cfr", "mccfr"}:
            raise ValueError("algo must be 'cfr' or 'mccfr'")
        self.iterations = iterations
        self.algo = algo

    def resolve(
        self, state: GameState, iterations: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Run a subgame solve starting from 'state' and return strategy for the
        acting player at the root.
        """
        if state.is_terminal():
            return {}

        total_iterations = iterations or self.iterations
        return resolve_subgame(state, total_iterations, algo=self.algo)


def resolve_subgame(state: GameState, iterations: int, algo: str = "mccfr") -> Dict[str, float]:
    """
    Helper that launches a fresh CFRTrainer from the provided state and
    computes the average strategy at the root infoset.
    """
    trainer = CFRTrainer()
    reach = [1.0] * state.num_players

    if algo == "cfr":
        for _ in range(iterations):
            trainer.cfr(state.clone(), list(reach))
    else:
        for _ in range(iterations):
            for traverser in range(state.num_players):
                trainer.cfr_sample(state.clone(), list(reach), traverser)

    actions = state.legal_actions()
    if not actions:
        return {}

    root_key = trainer._encode_state(state, state.to_act)  # type: ignore[attr-defined]
    node = trainer.node_map.get(root_key)
    if node is None:
        return _uniform_strategy(actions)

    avg = node.get_average_strategy()
    return {node.actions[i]: avg[i] for i in range(len(node.actions))}


def blend_strategies(
    blueprint: Dict[str, float],
    resolver: Dict[str, float],
    blueprint_weight: float,
) -> Dict[str, float]:
    """
    Mix two probability tables. blueprint_weight indicates how much mass to keep
    on the cached strategy (1.0 = ignore resolver, 0.0 = resolver only).
    """
    if not resolver:
        return blueprint

    weight = min(1.0, max(0.0, blueprint_weight))
    blended: Dict[str, float] = {}
    for action in set(blueprint) | set(resolver):
        base = blueprint.get(action, 0.0)
        adj = resolver.get(action, 0.0)
        blended[action] = weight * base + (1.0 - weight) * adj

    total = sum(blended.values())
    if total <= 0 or math.isclose(total, 0.0):
        return blueprint
    return {a: v / total for a, v in blended.items()}


def _uniform_strategy(actions: Dict[str, float] | list[str]) -> Dict[str, float]:
    if isinstance(actions, dict):
        keys = list(actions.keys())
    else:
        keys = list(actions)
    if not keys:
        return {}
    prob = 1.0 / len(keys)
    return {a: prob for a in keys}




