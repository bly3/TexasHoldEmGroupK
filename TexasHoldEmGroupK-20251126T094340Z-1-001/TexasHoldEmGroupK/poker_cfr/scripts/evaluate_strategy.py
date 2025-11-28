"""Evaluate a trained CFR strategy by simulating games."""

import argparse
import random
from typing import Dict, List

from poker_cfr.bot.agent import CFRAgent
from poker_cfr.cfr.infoset import (
    compute_blocker_features,
    compute_hand_bucket,
    encode_infoset,
)
from poker_cfr.env.game import GameState, initial_state


def play_game(agent0: CFRAgent, agent1: CFRAgent, stack_size: int, blinds: tuple, debug: bool = False) -> tuple:
    """
    Play a single game and return (player0_chip_change, player1_chip_change, actions_taken).
    """
    state = initial_state(stack_size=stack_size, blinds=blinds)
    actions_taken = []
    
    if debug:
        print(f"\nInitial state: stacks={state.stacks}, initial_stacks={state.initial_stacks}, pot={state.pot}")
    
    while not state.is_terminal():
        current = state.to_act
        legal_actions = state.legal_actions()
        
        if debug and len(actions_taken) < 5:
            print(f"  Player {current}, street={state.street}, stacks={state.stacks}, pot={state.pot}, legal={legal_actions}")
        
        # Get observation for current player
        observation = {
            "position": "IP" if current == 1 else "OOP",
            "street": state.street,
            "stack": state.stacks[current],
            "pot": state.pot,
            "hole_cards": state.hole_cards[current],
            "board": state.board,
            "history": state.history,
        }
        
        # Get action from appropriate agent
        agent = agent0 if current == 0 else agent1
        action = agent.act(observation, legal_actions)
        
        actions_taken.append((current, action))
        if debug and len(actions_taken) < 5:
            print(f"    Action: {action}")
        
        state = state.next_state(action)
        
        if debug and len(actions_taken) < 5:
            print(f"    After: stacks={state.stacks}, pot={state.pot}, terminal={state.is_terminal()}")
    
    chip_change_0 = state.stacks[0] - state.initial_stacks[0]
    chip_change_1 = state.stacks[1] - state.initial_stacks[1]
    
    if debug:
        print(f"Final: stacks={state.stacks}, initial_stacks={state.initial_stacks}")
        print(f"Chip changes: {chip_change_0}, {chip_change_1}")
        print(f"Total pot: {state.pot} (should be 0 if terminal)")
    
    return chip_change_0, chip_change_1, actions_taken


def evaluate_strategy(
    strategy_path: str = "poker_cfr/data/strategy.pkl",
    num_games: int = 1000,
    stack_size: int = 40,
    small_blind: int = 1,
    big_blind: int = 2,
) -> None:
    """Run evaluation matches between two CFR agents."""
    print(f"Loading strategy from {strategy_path}...")
    agent0 = CFRAgent(strategy_path, "CFRBot0")
    agent1 = CFRAgent(strategy_path, "CFRBot1")
    
    print(f"Playing {num_games} games (stack={stack_size}, blinds={small_blind}/{big_blind})...")
    
    total_chips_0 = 0
    total_chips_1 = 0
    wins_0 = 0
    wins_1 = 0
    ties = 0
    game_lengths = []
    
    for i in range(num_games):
        chip_change_0, chip_change_1, actions = play_game(
            agent0, agent1, stack_size, (small_blind, big_blind)
        )
        total_chips_0 += chip_change_0
        total_chips_1 += chip_change_1
        game_lengths.append(len(actions))
        
        if chip_change_0 > chip_change_1:
            wins_0 += 1
        elif chip_change_1 > chip_change_0:
            wins_1 += 1
        else:
            ties += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{num_games} games...")
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nGames played: {num_games}")
    print(f"Stack size: {stack_size} chips")
    print(f"Blinds: {small_blind}/{big_blind}")
    print(f"\nPlayer 0 (OOP/SB):")
    print(f"  Wins: {wins_0} ({100*wins_0/num_games:.1f}%)")
    print(f"  Total chips won: {total_chips_0}")
    print(f"  Average chips/game: {total_chips_0/num_games:.2f}")
    print(f"\nPlayer 1 (IP/BB):")
    print(f"  Wins: {wins_1} ({100*wins_1/num_games:.1f}%)")
    print(f"  Total chips won: {total_chips_1}")
    print(f"  Average chips/game: {total_chips_1/num_games:.2f}")
    print(f"\nTies: {ties} ({100*ties/num_games:.1f}%)")
    print(f"\nAverage game length: {sum(game_lengths)/len(game_lengths):.1f} actions")
    print(f"Total chips in play: {total_chips_0 + total_chips_1} (should be ~0)")
    print("\n" + "=" * 60)
    print("\nNOTE: Both players use the same strategy.")
    print("Expected: ~50/50 win rate (any deviation is due to randomness).")
    print("A well-trained CFR strategy should be balanced and unexploitable.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CFR strategy performance.")
    parser.add_argument("--strategy", type=str, default="poker_cfr/data/strategy.pkl", 
                       help="Path to strategy pickle file.")
    parser.add_argument("--games", type=int, default=1000, 
                       help="Number of games to simulate.")
    parser.add_argument("--stack", type=int, default=40, 
                       help="Stack size in chips.")
    parser.add_argument("--small-blind", type=int, default=1, 
                       help="Small blind amount.")
    parser.add_argument("--big-blind", type=int, default=2, 
                       help="Big blind amount.")
    args = parser.parse_args()
    
    evaluate_strategy(
        strategy_path=args.strategy,
        num_games=args.games,
        stack_size=args.stack,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
    )


if __name__ == "__main__":
    main()

