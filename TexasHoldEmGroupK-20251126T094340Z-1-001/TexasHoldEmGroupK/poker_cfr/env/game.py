"""
Multi-player No-Limit Hold'em abstraction (supports 2-6 players).

GameState supports standard Texas Hold'em betting with abstracted bet sizing
("fold", "call"/"check", "bet_small", "bet_pot", "bet_allin").
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import List, Tuple, Iterable, Optional, Set


Action = str  # e.g., "fold", "call", "bet_small", "bet_pot", "bet_allin"
STREETS = ["preflop", "flop", "turn", "river"]


@dataclass
class GameState:
    deck: List[str] = field(default_factory=list)
    hole_cards: List[List[str]] = field(default_factory=list)
    board: List[str] = field(default_factory=list)
    pot: int = 0  # includes current round contributions
    stacks: List[int] = field(default_factory=list)
    round_bets: List[int] = field(default_factory=list)  # contributions this street
    initial_stacks: List[int] = field(default_factory=list)
    current_player: int = 0
    num_players: int = 6  # Total number of players
    button: int = 0  # Button position
    active_players: Set[int] = field(default_factory=set)  # Players still in hand
    street: str = "preflop"  # preflop, flop, turn, river
    history: List[Action] = field(default_factory=list)
    terminal: bool = False
    winners: List[int] = field(default_factory=list)  # Can be multiple in multi-way pot
    consecutive_checks: int = 0
    last_aggressor: Optional[int] = None  # Last player to bet/raise

    def clone(self) -> "GameState":
        return GameState(
            deck=list(self.deck),
            hole_cards=[list(h) for h in self.hole_cards],
            board=list(self.board),
            pot=self.pot,
            stacks=list(self.stacks),
            round_bets=list(self.round_bets),
            initial_stacks=list(self.initial_stacks),
            current_player=self.current_player,
            num_players=self.num_players,
            button=self.button,
            active_players=set(self.active_players),
            street=self.street,
            history=list(self.history),
            terminal=self.terminal,
            winners=list(self.winners),
            consecutive_checks=self.consecutive_checks,
            last_aggressor=self.last_aggressor,
        )

    @property
    def to_act(self) -> int:
        return self.current_player

    def legal_actions(self) -> List[Action]:
        """
        Returns a list of abstract actions available at this state based on stack
        and outstanding bet.
        """
        if self.terminal or self.current_player not in self.active_players:
            return []
        to_call = self._to_call(self.current_player)
        actions: List[Action] = []
        actions.append("fold")
        actions.append("call")
        if self.stacks[self.current_player] > 0:
            actions.append("bet_small")
            actions.append("bet_pot")
            actions.append("bet_allin")
        return actions

    def is_terminal(self) -> bool:
        return self.terminal

    def is_chance_node(self) -> bool:
        # Chance is handled internally by dealing cards during transitions.
        return False

    def chance_outcomes(self) -> Iterable[Tuple[float, "GameState"]]:
        """
        Not used because we deal cards deterministically within transitions.
        """
        return []

    def next_state(self, action: Action) -> "GameState":
        """
        Apply action and return the resulting state with betting, street
        transitions, and showdown resolution.
        """
        next_s = self.clone()
        next_s.history.append(action)
        actor = self.current_player
        to_call = next_s._to_call(actor)

        if action == "fold":
            next_s.active_players.discard(actor)
            
            # If only one player left, award pot
            if len(next_s.active_players) == 1:
                next_s.terminal = True
                winner = list(next_s.active_players)[0]
                next_s.winners = [winner]
                next_s.stacks[winner] += next_s.pot
                next_s.pot = 0
                return next_s
            
            # Move to next active player
            next_s.current_player = next_s._next_active_player(actor)
            return next_s

        if action == "call":
            pay = min(to_call, next_s.stacks[actor])
            next_s._commit(actor, pay)
            if to_call == 0:
                next_s.consecutive_checks += 1
            else:
                next_s.consecutive_checks = 0

            # Check if betting round is complete
            # For multi-player: betting is complete if all active players have equal bets
            # and either: (a) no one has bet (all checked), or (b) action returned to last aggressor
            betting_done = next_s._betting_complete()
            
            if betting_done:
                next_s.consecutive_checks = 0
                if next_s._should_runout():
                    next_s._run_out_board()
                    next_s._resolve_showdown()
                    return next_s
                if next_s.street == "river":
                    next_s._resolve_showdown()
                    return next_s
                next_s._advance_street()
                # Start next street with first active player after button (or small blind if preflop)
                next_s.current_player = next_s._first_to_act_postflop()
            else:
                next_s.current_player = next_s._next_active_player(actor)
            return next_s

        if action == "bet_allin":
            pay = next_s.stacks[actor]
            next_s._commit(actor, pay)
            next_s.consecutive_checks = 0
            next_s.last_aggressor = actor
            if next_s._should_runout():
                next_s._run_out_board()
                next_s._resolve_showdown()
                return next_s
            next_s.current_player = next_s._next_active_player(actor)
            return next_s

        if action == "bet_small":
            pay = next_s._small_bet(actor, to_call)
            next_s._commit(actor, pay)
            next_s.consecutive_checks = 0
            next_s.last_aggressor = actor
            if next_s._should_runout():
                next_s._run_out_board()
                next_s._resolve_showdown()
                return next_s
            next_s.current_player = next_s._next_active_player(actor)
            return next_s

        if action == "bet_pot":
            pay = next_s._pot_sized_bet(actor, to_call)
            next_s._commit(actor, pay)
            next_s.consecutive_checks = 0
            next_s.last_aggressor = actor
            if next_s._should_runout():
                next_s._run_out_board()
                next_s._resolve_showdown()
                return next_s
            next_s.current_player = next_s._next_active_player(actor)
            return next_s

        # Unknown action: treat as check to be safe.
        next_s.current_player = next_s._next_active_player(actor)
        return next_s

    def utility(self, player: int) -> float:
        """
        Returns the payoff for 'player' at terminal nodes.
        """
        if not self.terminal:
            raise ValueError("utility called on non-terminal state")
        return float(self.stacks[player] - self.initial_stacks[player])
    
    def get_position(self, player: int) -> str:
        """
        Get position string for a player in multi-player game.
        In 6-player: UTG, MP, CO, BTN, SB, BB
        Simplified to: early (UTG/MP), middle (CO), late (BTN), blinds (SB/BB)
        """
        num_active = len(self.active_players)
        if num_active <= 2:
            # Heads-up: button is late, other is early
            if player == self.button:
                return "IP"  # Button acts last
            return "OOP"
        
        # Multi-way positions
        btn = self.button
        sb = (btn + 1) % self.num_players
        bb = (btn + 2) % self.num_players
        
        if player == btn:
            return "BTN"
        elif player == sb:
            return "SB"
        elif player == bb:
            return "BB"
        else:
            # UTG, MP, CO positions (roughly early/middle/late)
            positions = []
            for i in range(3, self.num_players):
                pos = (btn + i) % self.num_players
                if pos != btn and pos != sb and pos != bb:
                    positions.append(pos)
            
            if player in positions[:len(positions)//2]:
                return "EARLY"  # UTG/MP
            else:
                return "MIDDLE"  # CO

    # --- Internal helpers -------------------------------------------------

    def _advance_street(self) -> None:
        """Move to the next street and deal community cards."""
        idx = STREETS.index(self.street)
        if idx + 1 >= len(STREETS):
            self.street = "river"
            return
        self.street = STREETS[idx + 1]
        self.round_bets = [0] * self.num_players
        self.consecutive_checks = 0
        self.last_aggressor = None
        if self.street == "flop":
            self._deal_board(3)
        elif self.street == "turn":
            self._deal_board(1)
        elif self.street == "river":
            self._deal_board(1)

    def _deal_board(self, n: int) -> None:
        for _ in range(n):
            if self.deck:
                self.board.append(self.deck.pop())

    def _run_out_board(self) -> None:
        """If players are all-in, deal remaining community cards to river."""
        if len(self.board) < 3:
            self._deal_board(3 - len(self.board))
        if len(self.board) < 4:
            self._deal_board(4 - len(self.board))
        if len(self.board) < 5:
            self._deal_board(5 - len(self.board))
        self.street = "river"
        self.round_bets = [0] * self.num_players

    def _resolve_showdown(self) -> None:
        from poker_cfr.env.hand_eval import evaluate_multi_winner

        # Evaluate all active players' hands
        winners = evaluate_multi_winner(
            [self.hole_cards[p] for p in self.active_players],
            list(self.active_players),
            self.board
        )
        self.winners = winners
        
        # Split pot among winners
        if winners:
            split_per_winner = self.pot // len(winners)
            remainder = self.pot % len(winners)
            for i, winner in enumerate(winners):
                amount = split_per_winner + (1 if i < remainder else 0)
                self.stacks[winner] += amount
        
        self.pot = 0
        self.terminal = True

    def _commit(self, player: int, amount: int) -> None:
        self.stacks[player] -= amount
        self.pot += amount
        self.round_bets[player] += amount

    def _to_call(self, player: int) -> int:
        if not self.active_players:
            return 0
        max_bet = max(self.round_bets[p] for p in self.active_players if p < len(self.round_bets))
        return max(0, max_bet - self.round_bets[player])

    def _pot_sized_bet(self, player: int, to_call: int) -> int:
        """
        Compute a pot-sized wager (call + pot). Always at least 1 chip and
        capped by stack size.
        """
        desired = to_call + max(1, self.pot)
        return min(self.stacks[player], desired)

    def _small_bet(self, player: int, to_call: int) -> int:
        """
        Half-pot sizing approximation (call + 0.5*pot).
        """
        desired = to_call + max(1, self.pot // 2)
        return min(self.stacks[player], desired)

    def _should_runout(self) -> bool:
        """True if no further betting is possible (all remaining chips in)."""
        active_stacks = [self.stacks[p] for p in self.active_players]
        if not active_stacks:
            return True
        if min(active_stacks) > 0:
            return False
        # All active players have same round bet (all-in or checked)
        active_bets = [self.round_bets[p] for p in self.active_players if p < len(self.round_bets)]
        if not active_bets:
            return True
        return len(set(active_bets)) == 1
    
    def _next_active_player(self, current: int) -> int:
        """Get next active player in betting order."""
        if not self.active_players:
            return current
        for i in range(1, self.num_players):
            next_p = (current + i) % self.num_players
            if next_p in self.active_players:
                return next_p
        return current
    
    def _first_to_act_postflop(self) -> int:
        """First player to act postflop (small blind or first active after button)."""
        sb = (self.button + 1) % self.num_players
        if sb in self.active_players:
            return sb
        return self._next_active_player(self.button)
    
    def _betting_complete(self) -> bool:
        """Check if betting round is complete (all active players have equal bets and action is back to last aggressor)."""
        if len(self.active_players) <= 1:
            return True
        
        # All active players must have equal round bets
        active_bets = [self.round_bets[p] for p in self.active_players if p < len(self.round_bets)]
        if not active_bets:
            return True
        if len(set(active_bets)) > 1:
            return False
        
        # All players have equal bets. Now check if action has completed the round.
        
        # Case 1: No one has bet/raised (all checked or called blinds)
        if self.last_aggressor is None:
            # Betting is complete when all players have checked
            # consecutive_checks counts sequential checks/calls with no bet
            return self.consecutive_checks >= len(self.active_players)
        
        # Case 2: Someone bet/raised. Action must return to them.
        if self.last_aggressor not in self.active_players:
            # Aggressor folded, so betting is complete
            return True
        
        # Check if we've come full circle back to the aggressor
        if self.current_player == self.last_aggressor:
            return True
        
        # Check if all players after the aggressor have had a chance to act
        # and they've all called (no raises)
        # Simple heuristic: if consecutive_checks >= num_players - 1 and we're past aggressor
        active_list = list(self.active_players)
        if len(active_list) < 2:
            return True
        
        # Find aggressor position in betting order
        aggressor_pos = None
        for i in range(self.num_players):
            p = (self.last_aggressor + i) % self.num_players
            if p in active_list:
                aggressor_pos = active_list.index(p) if p in active_list else None
                break
        
        if aggressor_pos is not None:
            # Count how many players have acted since the aggressor
            # If all players after aggressor have acted and called, betting is complete
            players_after_aggressor = (len(active_list) - aggressor_pos - 1) % len(active_list)
            if self.consecutive_checks >= players_after_aggressor + 1:
                return True
        
        return False


def initial_state(
    stack_size: int = 40,
    blinds: Tuple[int, int] = (1, 2),
    num_players: int = 6
) -> GameState:
    """
    Create an initial state with stacks and blinds posted, shuffled deck, and
    hole cards dealt. Supports 2-6 players.
    """
    if num_players < 2 or num_players > 6:
        raise ValueError("num_players must be between 2 and 6")
    
    small_blind, big_blind = blinds
    deck = _new_deck()
    random.shuffle(deck)
    
    # Deal hole cards to all players
    hole = [[] for _ in range(num_players)]
    for _ in range(2):
        for i in range(num_players):
            hole[i].append(deck.pop())
    
    # Initialize stacks and blinds
    stacks = [stack_size] * num_players
    round_bets = [0] * num_players
    
    # Post blinds
    button = 0
    sb = (button + 1) % num_players
    bb = (button + 2) % num_players
    
    stacks[sb] -= small_blind
    stacks[bb] -= big_blind
    round_bets[sb] = small_blind
    round_bets[bb] = big_blind
    pot = small_blind + big_blind
    
    # First to act preflop is UTG (button + 3) or first active after BB
    if num_players > 2:
        first_act = (bb + 1) % num_players
    else:
        first_act = sb  # Heads-up: small blind acts first
    
    active_players = set(range(num_players))
    
    return GameState(
        deck=deck,
        hole_cards=hole,
        board=[],
        pot=pot,
        stacks=stacks,
        round_bets=round_bets,
        initial_stacks=[stack_size] * num_players,
        current_player=first_act,
        num_players=num_players,
        button=button,
        active_players=active_players,
        street="preflop",
        history=[],
        terminal=False,
        winners=[],
        consecutive_checks=0,
        last_aggressor=None,
    )


def _new_deck() -> List[str]:
    ranks = "23456789TJQKA"
    suits = "shdc"  # spades, hearts, diamonds, clubs
    return [r + s for r in ranks for s in suits]
