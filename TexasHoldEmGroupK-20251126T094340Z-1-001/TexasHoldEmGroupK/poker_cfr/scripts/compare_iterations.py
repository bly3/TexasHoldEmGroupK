"""
Compare CFR strategies trained with different iteration counts (or supplied
strategy files) by playing heads-up matches between every pair.
"""

from __future__ import annotations

import argparse
import itertools
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from poker_cfr.bot.agent import CFRAgent
from poker_cfr.cfr.storage import save_strategy
from poker_cfr.cfr.trainer import CFRTrainer
from poker_cfr.env.game import initial_state


@dataclass
class StrategySpec:
    label: str
    path: Path
    iterations: int | None = None
    source: str = "loaded"  # "trained" or "loaded"


@dataclass
class MatchStats:
    chips: float = 0.0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    games: int = 0
    game_lengths: List[int] = field(default_factory=list)

    def record_game(self, my_chips: float, opp_chips: float, num_actions: int) -> None:
        self.chips += my_chips
        self.games += 1
        if my_chips > opp_chips:
            self.wins += 1
        elif my_chips < opp_chips:
            self.losses += 1
        else:
            self.ties += 1
        self.game_lengths.append(num_actions)

    @property
    def avg_chips(self) -> float:
        return self.chips / self.games if self.games else 0.0

    @property
    def win_pct(self) -> float:
        return 100.0 * self.wins / self.games if self.games else 0.0

    @property
    def avg_length(self) -> float:
        if not self.game_lengths:
            return 0.0
        return sum(self.game_lengths) / len(self.game_lengths)


def play_game(
    agent_sb: CFRAgent,
    agent_bb: CFRAgent,
    stack_size: int,
    blinds: Tuple[int, int],
) -> Tuple[int, int, int]:
    """
    Play a single heads-up game. Returns (sb_chip_change, bb_chip_change, actions_taken).
    """
    state = initial_state(stack_size=stack_size, blinds=blinds)
    actions_taken = 0

    while not state.is_terminal():
        current = state.to_act
        legal_actions = state.legal_actions()
        if not legal_actions:
            break

        observation = {
            "position": "IP" if current == 1 else "OOP",
            "street": state.street,
            "stack": state.stacks[current],
            "pot": state.pot,
            "hole_cards": state.hole_cards[current],
            "board": state.board,
            "history": state.history,
        }

        agent = agent_sb if current == 0 else agent_bb
        action = agent.act(observation, legal_actions)

        state = state.next_state(action)
        actions_taken += 1

    chip_change_sb = state.stacks[0] - state.initial_stacks[0]
    chip_change_bb = state.stacks[1] - state.initial_stacks[1]
    return chip_change_sb, chip_change_bb, actions_taken


def run_head_to_head(
    spec_a: StrategySpec,
    spec_b: StrategySpec,
    total_games: int,
    stack_size: int,
    blinds: Tuple[int, int],
    swap_seats: bool = True,
) -> Dict[str, MatchStats]:
    stats = {spec_a.label: MatchStats(), spec_b.label: MatchStats()}

    if total_games <= 0:
        return stats

    if swap_seats:
        first_games = total_games // 2 + (total_games % 2)
        second_games = total_games // 2
        orientations = [
            (spec_a, spec_b, first_games),
            (spec_b, spec_a, second_games),
        ]
    else:
        orientations = [(spec_a, spec_b, total_games)]

    for sb_spec, bb_spec, games in orientations:
        if games <= 0:
            continue

        agent_sb = CFRAgent(str(sb_spec.path), name=f"{sb_spec.label}-SB")
        agent_bb = CFRAgent(str(bb_spec.path), name=f"{bb_spec.label}-BB")

        for _ in range(games):
            chip_sb, chip_bb, actions = play_game(agent_sb, agent_bb, stack_size, blinds)
            stats[sb_spec.label].record_game(chip_sb, chip_bb, actions)
            stats[bb_spec.label].record_game(chip_bb, chip_sb, actions)

    return stats


def accumulate(dest: MatchStats, src: MatchStats) -> None:
    dest.chips += src.chips
    dest.wins += src.wins
    dest.losses += src.losses
    dest.ties += src.ties
    dest.games += src.games
    dest.game_lengths.extend(src.game_lengths)


def format_match_summary(label_a: str, label_b: str, stats: Dict[str, MatchStats]) -> str:
    a_stats = stats[label_a]
    b_stats = stats[label_b]
    total_games = a_stats.games
    avg_len = (a_stats.avg_length + b_stats.avg_length) / 2.0 if total_games else 0.0

    lines = [
        f"\n=== {label_a} vs {label_b} ===",
        f"Games: {total_games}",
        f"Average actions/game: {avg_len:.1f}",
        f"{label_a}: avg chips {a_stats.avg_chips:.2f}, win% {a_stats.win_pct:.1f} "
        f"(W/L/T {a_stats.wins}/{a_stats.losses}/{a_stats.ties})",
        f"{label_b}: avg chips {b_stats.avg_chips:.2f}, win% {b_stats.win_pct:.1f} "
        f"(W/L/T {b_stats.wins}/{b_stats.losses}/{b_stats.ties})",
    ]
    return "\n".join(lines)


def train_strategies(
    iterations: Iterable[int],
    stack_size: int,
    blinds: Tuple[int, int],
    algo: str,
    cores: int,
    num_players: int,
    output_dir: Path,
) -> List[StrategySpec]:
    specs: List[StrategySpec] = []
    for iters in iterations:
        print(f"\n--- Training strategy with {iters:,} iterations ---")

        trainer = CFRTrainer()

        def state_factory():
            return initial_state(
                stack_size=stack_size,
                blinds=blinds,
                num_players=num_players,
            )

        strategy = trainer.train(
            iterations=iters,
            initial_state_factory=state_factory,
            algo=algo,
            train_player=0,
            num_cores=cores,
            stack_size=stack_size,
            blinds=blinds,
        )

        filename = f"strategy_{iters}_iters.pkl"
        output_path = output_dir / filename
        save_strategy(output_path, strategy)
        specs.append(
            StrategySpec(
                label=f"{iters:,} iters",
                path=output_path,
                iterations=iters,
                source="trained",
            )
        )
        print(f"Saved strategy to {output_path}")
    return specs


def parse_strategy_args(args: Sequence[str]) -> List[StrategySpec]:
    specs: List[StrategySpec] = []
    for entry in args:
        if "=" not in entry:
            raise ValueError(
                f"Invalid strategy format '{entry}'. Use label=path (e.g., 5k=data/strategy_5000.pkl)."
            )
        label, path = entry.split("=", 1)
        full_path = Path(path).expanduser()
        if not full_path.exists():
            raise FileNotFoundError(f"Strategy path does not exist: {full_path}")
        specs.append(StrategySpec(label=label, path=full_path))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and compare CFR strategies with different iteration counts."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="*",
        default=[],
        help="Iteration counts to train from scratch (e.g., --iterations 5000 100000).",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="*",
        default=[],
        help="Existing strategies to include, format label=path.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=400,
        help="Total heads-up games per matchup (automatically split across seats).",
    )
    parser.add_argument("--stack", type=int, default=40, help="Stack size in chips.")
    parser.add_argument("--small-blind", type=int, default=1, help="Small blind.")
    parser.add_argument("--big-blind", type=int, default=2, help="Big blind.")
    parser.add_argument(
        "--algo",
        choices=["cfr", "mccfr"],
        default="cfr",
        help="Training algorithm for newly trained strategies.",
    )
    parser.add_argument("--cores", type=int, default=1, help="CPU cores for training.")
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of players for training (head-to-head evaluation requires 2).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="poker_cfr/data/experiments",
        help="Directory to store newly trained strategy files.",
    )
    parser.add_argument(
        "--keep-strategies",
        action="store_true",
        help="Keep newly trained strategy files on disk instead of deleting them afterwards.",
    )
    parser.add_argument(
        "--disable-seat-swap",
        action="store_true",
        help="Do not swap seats during matches (default: swap).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible training/evaluation.",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.num_players != 2:
        raise ValueError("Head-to-head evaluation currently supports num_players=2.")

    blinds = (args.small_blind, args.big_blind)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trained_specs = train_strategies(
        args.iterations,
        stack_size=args.stack,
        blinds=blinds,
        algo=args.algo,
        cores=args.cores,
        num_players=args.num_players,
        output_dir=output_dir,
    )
    loaded_specs = parse_strategy_args(args.strategies)
    all_specs = trained_specs + loaded_specs

    if len(all_specs) < 2:
        raise ValueError(
            "Need at least two strategies (trained or loaded) to run a comparison."
        )

    overall: Dict[str, MatchStats] = {spec.label: MatchStats() for spec in all_specs}
    seat_swap = not args.disable-seat_swap

    for spec_a, spec_b in itertools.combinations(all_specs, 2):
        print(
            f"\nRunning matchup {spec_a.label} vs {spec_b.label} "
            f"({args.games} games total, seat swap={'on' if seat_swap else 'off'})"
        )
        pair_stats = run_head_to_head(
            spec_a,
            spec_b,
            total_games=args.games,
            stack_size=args.stack,
            blinds=blinds,
            swap_seats=seat_swap,
        )

        for label, stats in pair_stats.items():
            accumulate(overall[label], stats)

        print(format_match_summary(spec_a.label, spec_b.label, pair_stats))

    leaderboard = sorted(
        overall.items(), key=lambda item: item[1].avg_chips, reverse=True
    )

    print("\n" + "=" * 70)
    print("LEADERBOARD (sorted by avg chips per game across all matchups)")
    print("=" * 70)
    header = f"{'Label':<20} {'Games':>8} {'Win%':>8} {'Avg Chips':>12} {'W/L/T':>12}"
    print(header)
    for label, stats in leaderboard:
        print(
            f"{label:<20} {stats.games:>8} {stats.win_pct:>7.1f}% "
            f"{stats.avg_chips:>12.2f} "
            f"{stats.wins}/{stats.losses}/{stats.ties}"
        )

    generated_paths = [spec.path for spec in trained_specs]
    if generated_paths and not args.keep_strategies:
        for path in generated_paths:
            try:
                path.unlink()
                print(f"Deleted temporary strategy {path}")
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()

