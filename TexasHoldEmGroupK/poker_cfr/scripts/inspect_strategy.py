"""Print a few infosets from a saved strategy."""

import argparse

from poker_cfr.cfr.storage import load_strategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect saved strategy.")
    parser.add_argument("--path", type=str, default="poker_cfr/data/strategy.pkl")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    strategy = load_strategy(args.path)
    print(f"Loaded strategy with {len(strategy)} infosets")
    for i, (infoset, probs) in enumerate(strategy.items()):
        if i >= args.n:
            break
        print(infoset, "->", probs)


if __name__ == "__main__":
    main()
