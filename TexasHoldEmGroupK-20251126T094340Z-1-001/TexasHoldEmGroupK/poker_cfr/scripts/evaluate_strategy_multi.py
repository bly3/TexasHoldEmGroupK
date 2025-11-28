"""Evaluate a trained CFR strategy in multiplayer games (2-6 players)."""

import argparse
from typing import Dict, List
import random

from poker_cfr.bot.agent import CFRAgent
from poker_cfr.env.game import initial_state


def play_multiplayer_game(
    agents: List[CFRAgent],
    stack_size: int,
    blinds: tuple,
    num_players: int,
    debug: bool = False
) -> tuple:
    """Play a single multiplayer game and return chip changes for all players."""
    state = initial_state(stack_size=stack_size, blinds=blinds, num_players=num_players)
    actions_taken = []
    
    if debug:
        print(f"\nInitial state: {num_players} players, stacks={state.stacks}, pot={state.pot}")
    
    while not state.is_terminal():
        current = state.to_act
        legal_actions = state.legal_actions()
        
        if not legal_actions or current not in state.active_players:
            break
        
        if debug and len(actions_taken) < 10:
            print(f"  Player {current} ({state.get_position(current)}), street={state.street}, "
                  f"active={len(state.active_players)}, pot={state.pot}, legal={legal_actions}")
        
        # Get observation for current player
        observation = {
            "position": state.get_position(current),
            "street": state.street,
            "stack": state.stacks[current],
            "pot": state.pot,
            "hole_cards": state.hole_cards[current],
            "board": state.board,
            "history": state.history,
        }
        
        # Get action from appropriate agent
        agent = agents[current]
        action = agent.act(observation, legal_actions)
        
        actions_taken.append((current, action))
        if debug and len(actions_taken) < 10:
            print(f"    Action: {action}")
        
        state = state.next_state(action)
    
    # Calculate chip changes for all players
    chip_changes = [
        state.stacks[i] - state.initial_stacks[i]
        for i in range(num_players)
    ]
    
    if debug:
        print(f"Final stacks: {state.stacks}")
        print(f"Chip changes: {chip_changes}")
        print(f"Winners: {state.winners}")
    
    return chip_changes, actions_taken


def evaluate_multiplayer_strategy(
    strategy_path: str = "poker_cfr/data/strategy.pkl",
    num_games: int = 500,
    stack_size: int = 40,
    small_blind: int = 1,
    big_blind: int = 2,
    num_players: int = 6,
) -> None:
    """Evaluate strategy in multiplayer games."""
    print(f"Loading strategy from {strategy_path}...")
    
    # Create agents for all players (all using the same strategy)
    agents = [CFRAgent(strategy_path, f"CFRBot{i}") for i in range(num_players)]
    
    print(f"Playing {num_games} {num_players}-player games (stack={stack_size}, blinds={small_blind}/{big_blind})...")
    
    # Track stats per player
    total_chips = [0] * num_players
    wins = [0] * num_players
    ties = [0] * num_players
    game_lengths = []
    
    for i in range(num_games):
        chip_changes, actions = play_multiplayer_game(
            agents, stack_size, (small_blind, big_blind), num_players
        )
        
        for p in range(num_players):
            total_chips[p] += chip_changes[p]
        
        game_lengths.append(len(actions))
        
        # Determine winners (players with max chip change)
        max_chips = max(chip_changes)
        winners = [p for p, chips in enumerate(chip_changes) if chips == max_chips]
        
        if len(winners) == 1:
            wins[winners[0]] += 1
        else:
            for w in winners:
                ties[w] += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{num_games} games...")
    
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS ({num_players}-PLAYER GAMES)")
    print("=" * 70)
    print(f"\nGames played: {num_games}")
    print(f"Players: {num_players}")
    print(f"Stack size: {stack_size} chips")
    print(f"Blinds: {small_blind}/{big_blind}")
    
    print(f"\n{'Player':<8} {'Wins':<8} {'Win%':<8} {'Avg Chips':<12} {'Total Chips':<12}")
    print("-" * 70)
    
    for p in range(num_players):
        win_pct = 100 * wins[p] / num_games
        avg_chips = total_chips[p] / num_games
        print(f"{p:<8} {wins[p]:<8} {win_pct:<7.1f}% {avg_chips:<12.2f} {total_chips[p]:<12}")
    
    print(f"\nAverage game length: {sum(game_lengths)/len(game_lengths):.1f} actions")
    total_all_chips = sum(total_chips)
    print(f"Total chips in play: {total_all_chips} (should be ~0)")
    
    # Expected win rate for fair strategy: ~1/num_players for each player
    expected_win_rate = 100 / num_players
    print(f"\nExpected win rate per player: {expected_win_rate:.1f}%")
    print("\n" + "=" * 70)
    print("\nNOTE: All players use the same strategy.")
    print("In multiplayer, a fair strategy should win ~1/N of the time per player.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CFR strategy in multiplayer games.")
    parser.add_argument("--strategy", type=str, default="poker_cfr/data/strategy.pkl", 
                       help="Path to strategy pickle file.")
    parser.add_argument("--games", type=int, default=500, 
                       help="Number of games to simulate.")
    parser.add_argument("--stack", type=int, default=40, 
                       help="Stack size in chips.")
    parser.add_argument("--small-blind", type=int, default=1, 
                       help="Small blind amount.")
    parser.add_argument("--big-blind", type=int, default=2, 
                       help="Big blind amount.")
    parser.add_argument("--num-players", type=int, default=6,
                       help="Number of players (2-6).")
    args = parser.parse_args()
    
    if args.num_players < 2 or args.num_players > 6:
        raise ValueError("num_players must be between 2 and 6")
    
    evaluate_multiplayer_strategy(
        strategy_path=args.strategy,
        num_games=args.games,
        stack_size=args.stack,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        num_players=args.num_players,
    )


if __name__ == "__main__":
    main()

