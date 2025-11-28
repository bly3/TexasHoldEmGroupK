"""
Async client for wiring CFR bots into the Go infrastructure server.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import websockets
from websockets import ConnectionClosed

from poker_cfr.bot.agent import CFRAgent

SUIT_LOOKUP = {
    "HEART": "h",
    "DIAMOND": "d",
    "CLUB": "c",
    "SPADE": "s",
}

DEFAULT_LEGAL_ACTIONS: Tuple[str, ...] = ("fold", "call", "bet_pot", "bet_allin")


@dataclass
class InfrastructureConfig:
    """
    Connection + runtime options for the infrastructure server.
    """

    server_url: str = "ws://localhost:8080/ws"
    api_key: str = "dev"
    table_id: str = "table-1"
    start_key: Optional[str] = None
    reconnect_delay: float = 2.0
    max_reconnect_delay: float = 15.0
    auto_start: bool = True
    min_players_to_start: int = 2
    bot_name_prefix: str = "CFRBot"

    def make_url(self, player_id: str, is_host: bool = False) -> str:
        query = f"apiKey={self.api_key}&table={self.table_id}&player={player_id}"
        if is_host and self.start_key:
            query += f"&startKey={self.start_key}"
        sep = "&" if "?" in self.server_url else "?"
        return f"{self.server_url}{sep}{query}"


class InfrastructureBotClient:
    """
    Wraps a CFRAgent and drives it via the WebSocket protocol exposed
    by the Go infrastructure server.
    """

    def __init__(
        self,
        player_id: str,
        agent: CFRAgent,
        config: InfrastructureConfig,
        *,
        is_host: bool = False,
        legal_actions: Sequence[str] = DEFAULT_LEGAL_ACTIONS,
    ) -> None:
        self.player_id = player_id
        self.agent = agent
        self.config = config
        self.is_host = bool(is_host and config.start_key)
        self.legal_actions = list(legal_actions)
        self.logger = logging.getLogger(f"infra.{player_id}")
        self._history: List[str] = []
        self._current_phase: Optional[str] = None
        self._last_decision_key: Optional[Tuple[Any, ...]] = None
        self._start_requested = False
        self._running = True

    async def run_forever(self) -> None:
        """
        Persistently connect to the table, auto-reconnecting with backoff.
        """
        delay = self.config.reconnect_delay
        while self._running:
            try:
                await self._run_once()
                delay = self.config.reconnect_delay
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - network heavy
                self.logger.warning("connection error (%s), retrying in %.1fs", exc, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, self.config.max_reconnect_delay)

    async def _run_once(self) -> None:
        url = self.config.make_url(self.player_id, self.is_host)
        self.logger.info("connecting to %s", url)
        async with websockets.connect(url, ping_interval=15, ping_timeout=10) as ws:
            await ws.send(json.dumps({"type": "join"}))
            self.logger.info("joined table %s", self.config.table_id)
            await self._listen(ws)

    async def _listen(self, ws: websockets.WebSocketClientProtocol) -> None:
        async for raw in ws:
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                self.logger.debug("invalid json: %s", raw)
                continue
            await self._handle_message(ws, message)

    async def _handle_message(
        self, ws: websockets.WebSocketClientProtocol, message: Dict[str, Any]
    ) -> None:
        msg_type = message.get("type")
        if msg_type == "state":
            state = message.get("state") or {}
            await self._handle_state(ws, state)
        elif msg_type == "error":
            self.logger.warning("server error: %s", message.get("error") or message)

    async def _handle_state(
        self, ws: websockets.WebSocketClientProtocol, state: Dict[str, Any]
    ) -> None:
        table = state.get("table") or {}
        phase = str(state.get("phase") or table.get("phase") or "").upper() or None

        if phase != self._current_phase:
            if phase == "PREFLOP":
                self._history = []
            self._current_phase = phase
            self._last_decision_key = None
        if self.is_host and self.config.auto_start:
            await self._maybe_start_hand(ws, table, phase)

        if phase == "WAITING":
            self._last_decision_key = None
            return

        seat_idx, player = self._find_player(table)
        if seat_idx is None or player is None:
            return

        to_act = state.get("toActIdx")
        if to_act != seat_idx:
            self._last_decision_key = None
            return

        decision_key = (
            phase,
            state.get("pot"),
            player.get("chips"),
            len(state.get("board") or table.get("cardOpen") or []),
        )
        if decision_key == self._last_decision_key:
            return
        self._last_decision_key = decision_key

        observation = self._build_observation(state, table, player, seat_idx)
        action = self.agent.act(observation, self.legal_actions)
        await self._send_action(ws, action, player, state)

    async def _maybe_start_hand(
        self,
        ws: websockets.WebSocketClientProtocol,
        table: Dict[str, Any],
        phase: Optional[str],
    ) -> None:
        if phase != "WAITING":
            self._start_requested = False
            return
        if self._start_requested:
            return
        players = [p for p in (table.get("players") or []) if p and p.get("chips", 0) > 0]
        if len(players) < self.config.min_players_to_start:
            return
        await ws.send(json.dumps({"type": "start_hand"}))
        self._start_requested = True
        self.logger.info("requested new hand start")

    def _find_player(self, table: Dict[str, Any]) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
        players = table.get("players") or []
        for idx, player in enumerate(players):
            if not player:
                continue
            if player.get("id") == self.player_id:
                return idx, player
        return None, None

    def _build_observation(
        self,
        state: Dict[str, Any],
        table: Dict[str, Any],
        player: Dict[str, Any],
        seat_idx: int,
    ) -> Dict[str, Any]:
        board = state.get("board") or table.get("cardOpen") or []
        observation = {
            "position": self._estimate_position(table, seat_idx, state.get("toActIdx")),
            "street": str((table.get("phase") or state.get("phase") or "preflop")).
            lower(),
            "stack": player.get("chips", 0),
            "pot": state.get("pot", 1),
            "hole_cards": _cards_to_strings(player.get("cards") or []),
            "board": _cards_to_strings(board),
            "history": list(self._history),
        }
        return observation

    def _estimate_position(
        self, table: Dict[str, Any], seat_idx: int, to_act_idx: Optional[int]
    ) -> str:
        players = [p for p in (table.get("players") or []) if p]
        if len(players) <= 2 and to_act_idx is not None:
            return "IP" if seat_idx != to_act_idx else "OOP"
        labels = ["SB", "BB", "UTG", "MP", "CO", "BTN"]
        if 0 <= seat_idx < len(labels):
            return labels[seat_idx]
        return "IP"

    async def _send_action(
        self,
        ws: websockets.WebSocketClientProtocol,
        action: str,
        player: Dict[str, Any],
        state: Dict[str, Any],
    ) -> None:
        verb, amount = self._map_action(action, player, state)
        payload: Dict[str, Any] = {
            "type": "act",
            "playerId": self.player_id,
            "action": verb,
        }
        if amount is not None:
            payload["amount"] = amount
        try:
            await ws.send(json.dumps(payload))
            self._history.append(action)
            self.logger.debug("sent %s", payload)
        except ConnectionClosed:
            self.logger.debug("connection closed before action sent")

    def _map_action(
        self, action: str, player: Dict[str, Any], state: Dict[str, Any]
    ) -> Tuple[str, Optional[int]]:
        normalized = action.lower()
        stack = int(player.get("chips", 0) or 0)
        pot = int(state.get("pot", 0) or 0)

        if normalized == "fold":
            return "FOLD", None
        if normalized == "call":
            return "CALL", None
        if normalized in {"bet_pot", "bet_small", "bet_allin"}:
            if stack <= 0:
                return "CALL", None
            if normalized == "bet_allin":
                return "RAISE", stack
            if normalized == "bet_pot":
                raise_size = max(1, min(stack, pot or 1))
            else:  # bet_small
                raise_size = max(1, min(stack, max(1, pot // 2)))
            if raise_size <= 0:
                return "CALL", None
            return "RAISE", raise_size
        # fallback for unknown actions
        return "CALL", None

    def stop(self) -> None:
        self._running = False


def _cards_to_strings(cards: Sequence[Any]) -> List[str]:
    formatted: List[str] = []
    for card in cards:
        if not card:
            continue
        rank = str(card.get("rank") or "").upper()
        suit = SUIT_LOOKUP.get(str(card.get("suit") or "").upper())
        if not rank or not suit:
            continue
        formatted.append(f"{rank}{suit}")
    return formatted


