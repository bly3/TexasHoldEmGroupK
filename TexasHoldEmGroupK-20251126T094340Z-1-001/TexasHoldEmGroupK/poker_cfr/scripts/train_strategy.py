"""Run CFR training and save the resulting strategy."""

import argparse
from pathlib import Path

from poker_cfr.cfr.storage import save_strategy
from poker_cfr.cfr.trainer import CFRTrainer
from poker_cfr.env.game import initial_state


def main() -> None:
    import multiprocessing
    
    parser = argparse.ArgumentParser(description="Train CFR/MCCFR strategy.")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations per player.")
    parser.add_argument("--stack", type=int, default=40, help="Stack size in chips.")
    parser.add_argument("--small-blind", type=int, default=1, help="Small blind amount.")
    parser.add_argument("--big-blind", type=int, default=2, help="Big blind amount.")
    parser.add_argument("--output", type=str, default="poker_cfr/data/strategy.pkl", help="Output pickle path.")
    parser.add_argument("--algo", choices=["cfr", "mccfr"], default="cfr", help="Training algorithm.")
    parser.add_argument("--num-players", type=int, default=6, help="Number of players (2-6).")
    parser.add_argument("--cores", type=int, default=1, help="Number of CPU cores to use for parallel training (default: 1 = sequential).")
    args = parser.parse_args()

    if args.num_players < 2 or args.num_players > 6:
        raise ValueError("num_players must be between 2 and 6")
    
    if args.cores < 1:
        raise ValueError("cores must be at least 1")
    
    max_cores = multiprocessing.cpu_count()
    if args.cores > max_cores:
        print(f"Warning: Requested {args.cores} cores but only {max_cores} available. Using {max_cores} cores.")
        args.cores = max_cores

    def state_factory():
        return initial_state(
            stack_size=args.stack,
            blinds=(args.small_blind, args.big_blind),
            num_players=args.num_players
        )

    if args.cores > 1:
        print(f"Training with {args.cores} CPU cores...")
    
    trainer = CFRTrainer()
    # Train player 0 (hero) against the other players
    # Pass game parameters for parallel training
    strategy = trainer.train(
        args.iterations, 
        state_factory, 
        algo=args.algo, 
        train_player=0, 
        num_cores=args.cores,
        stack_size=args.stack,
        blinds=(args.small_blind, args.big_blind)
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    try:
        save_strategy(args.output, strategy)
        print(f"[OK] Saved strategy with {len(strategy)} infosets to {args.output}")
    except Exception as e:
        print(f"[ERR] Error saving strategy: {e}")
        raise


if __name__ == "__main__":
    main()
