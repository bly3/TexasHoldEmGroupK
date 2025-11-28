"""
CFRAgent: loads a precomputed strategy, applies small opponent/style adjustments,
and returns a legal action.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence

from poker_cfr.bot.opponent_model import OpponentModel
from poker_cfr.cfr.infoset import (
    compute_blocker_features,
    compute_hand_bucket,
    encode_infoset,
)
from poker_cfr.cfr.storage import load_strategy
from poker_cfr.resolver.realtime import RealtimeResolver, blend_strategies
from poker_cfr.env.game import STREETS


class CFRAgent:
    def __init__(
        self,
        strategy_path: str = "poker_cfr/data/strategy.pkl",
        name: str = "CFRBot",
        play_style: str = "base",  # base|aggressive|nitty
        enable_resolver: bool = False,
        resolver_iterations: int = 600,
        resolver_algo: str = "mccfr",
        resolver_blueprint_weight: float = 0.55,
        resolver_trigger_street: str = "turn",
    ):
        self.name = name
        try:
            self.strategy: Dict[str, Dict[str, float]] = load_strategy(strategy_path)
        except FileNotFoundError:
            self.strategy = {}
        self.opponent_model = OpponentModel()
        self.play_style = play_style
        self.resolver = (
            RealtimeResolver(iterations=resolver_iterations, algo=resolver_algo)
            if enable_resolver
            else None
        )
        self.resolver_blueprint_weight = max(0.0, min(1.0, resolver_blueprint_weight))
        self.resolver_trigger_street = resolver_trigger_street.lower()

    def act(self, observation: Dict, legal_actions: Sequence[str]) -> str:
        """
        observation contains: hole_cards, board, history, position, street, stack, pot.
        """
        infoset_payload = _encode_from_observation(observation)
        infoset_key = encode_infoset(**infoset_payload)
        probs = self.strategy.get(infoset_key)

        if probs:
            probs = _align_probs_to_actions(probs, legal_actions)
            probs = adjust_probs_for_style_and_opp(
                probs, legal_actions, style=self.play_style, opp_model=self.opponent_model
            )
        else:
            probs = {a: 1.0 / len(legal_actions) for a in legal_actions}

        if self.resolver and _should_resolve(observation, self.resolver_trigger_street):
            resolver_state = observation.get("game_state")
            if resolver_state:
                resolver_probs = self.resolver.resolve(resolver_state)
                resolver_probs = _align_probs_to_actions(resolver_probs, legal_actions)
                probs = blend_strategies(
                    probs, resolver_probs, self.resolver_blueprint_weight
                )

        return _sample_action(probs)

    def observe(self, outcome: Dict) -> None:
        """
        Hook to feed results into the opponent model. Outcome can include
        action_history and showdown info.
        """
        self.opponent_model.update(outcome)


def _encode_from_observation(observation: Dict) -> Dict:
    hole = observation.get("hole_cards", [])
    board = observation.get("board", [])
    history = observation.get("history", [])
    blockers = observation.get("blockers") or compute_blocker_features(hole, board)
    hand_bucket = observation.get("hand_bucket") or compute_hand_bucket(hole, board)

    return dict(
        position=observation.get("position", "IP"),
        street=observation.get("street", "preflop"),
        stack=observation.get("stack", 0),
        pot=observation.get("pot", 1),
        hand_bucket=hand_bucket,
        board=board,
        blockers=blockers,
        history=history,
    )


def _align_probs_to_actions(probs: Dict[str, float], legal_actions: Sequence[str]) -> Dict[str, float]:
    aligned = {}
    missing = []
    for a in legal_actions:
        if a in probs:
            aligned[a] = probs[a]
        else:
            missing.append(a)
    # distribute any missing uniformly
    if missing:
        filler = 1.0 / (len(missing) + 1e-9)
        for a in missing:
            aligned[a] = filler
    total = sum(aligned.values())
    return {k: v / total for k, v in aligned.items()}


def adjust_probs_for_style_and_opp(
    probs: Dict[str, float],
    legal_actions: Sequence[str],
    style: str,
    opp_model: OpponentModel,
) -> Dict[str, float]:
    """
    Small heuristic tweaks:
    - aggressive: bump bet_pot/bet_allin slightly, reduce fold.
    - nitty: bump fold/call, reduce big bets.
    - base: no change.
    - Opponent model (if overfolding): increase aggression; if calling station: decrease bluffs.
    """
    adjusted = probs.copy()
    overfold = opp_model.is_overfolding()
    calling_station = opp_model.is_calling_station()

    def bump(action: str, factor: float) -> None:
        if action in adjusted:
            adjusted[action] *= factor

    if style == "aggressive":
        bump("bet_pot", 1.1)
        bump("bet_allin", 1.1)
        bump("fold", 0.9)
    elif style == "nitty":
        bump("fold", 1.1)
        bump("call", 1.05)
        bump("bet_allin", 0.9)

    if overfold:
        bump("bet_pot", 1.1)
        bump("bet_allin", 1.1)
    if calling_station:
        bump("bet_pot", 0.9)
        bump("bet_allin", 0.9)

    # renormalize
    total = sum(adjusted.get(a, 0.0) for a in legal_actions)
    if total <= 0:
        return {a: 1.0 / len(legal_actions) for a in legal_actions}
    return {a: adjusted.get(a, 0.0) / total for a in legal_actions}


_STREET_INDEX = {name: idx for idx, name in enumerate(STREETS)}


def _should_resolve(observation: Dict, trigger_street: str) -> bool:
    if not observation.get("game_state"):
        return False
    if observation.get("force_resolve"):
        return True
    current = str(observation.get("street", "preflop")).lower()
    return _STREET_INDEX.get(current, 0) >= _STREET_INDEX.get(trigger_street, 0)


def _sample_action(probs: Dict[str, float]) -> str:
    r = random.random()
    cumulative = 0.0
    items = list(probs.items())
    for action, p in items:
        cumulative += p
        if r <= cumulative:
            return action
    return items[-1][0]

