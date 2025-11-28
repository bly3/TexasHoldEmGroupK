# Poker CFR/MCCFR Bot Skeleton

This repository scaffolds a two-part project:
1) An offline training system using CFR/MCCFR over a simplified 2-6 player NLHE abstraction.
2) A tournament bot that loads the precomputed strategy and returns actions via infoset lookups.

## Quick Start

1. *(Optional)* Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate        # Windows
   source .venv/bin/activate       # macOS/Linux
   pip install -r requirements.txt
   ```
2. Train a strategy profile (all seats learn simultaneously because CFR updates the acting player's regrets):
   ```bash
   # 6-player MCCFR (recommended for ring games)
   python -m poker_cfr.scripts.train_strategy --iterations 1000 --num-players 6 --algo mccfr --output poker_cfr/data/strategy.pkl

   # Heads-up full CFR
   python -m poker_cfr.scripts.train_strategy --iterations 10000 --num-players 2 --algo cfr

   # Use multiple CPU cores for faster training (e.g., 16 cores on your machine, or 92 on EC2)
   python -m poker_cfr.scripts.train_strategy --iterations 10000 --num-players 6 --algo mccfr --cores 16
   ```
   Each iteration deals a fresh game via `poker_cfr.env.game.initial_state`, traverses the tree, and writes the averaged strategy table to the output pickle.
3. Evaluate the saved profile:
   ```bash
   # Heads-up
   python -m poker_cfr.scripts.evaluate_strategy --strategy poker_cfr/data/strategy.pkl --games 1000

   # Multiplayer (2-6 players)
   python -m poker_cfr.scripts.evaluate_strategy_multi --strategy poker_cfr/data/strategy.pkl --games 500 --num-players 6
   ```
   All seats use the exact same strategy during evaluation, so the results should be roughly break-even (each seat wins about 1 / num_players of the time).
4. Validate that more training actually helps by running controlled head-to-head arenas:
   ```bash
   python -m poker_cfr.scripts.compare_iterations \
       --iterations 5000 100000 \
       --games 600 \
       --stack 40 --small-blind 1 --big-blind 2
   ```
   The script can also include existing files via `--strategies label=path`, auto-swaps seats so both bots play SB/BB, reports per-match summaries, and prints a leaderboard sorted by average chips won. Use `--keep-strategies` if you want to keep the newly trained pickle files instead of deleting them afterward.
5. Integrate the bot elsewhere by instantiating `poker_cfr.bot.agent.CFRAgent` with the saved pickle path. Feed it infosets built with the same helpers (see `USAGE.md`).

## How CFR Training Works

**Key insight**: CFR trains one joint strategy profile that learns how to play every seat. You do not train separate bots for each player.

- During traversal, CFR updates regrets for whoever is acting at the node.
- With the multi-player extension, we track one reach probability per seat, so six players can learn together in a single run.
- Infoset encoding captures position, stack sizes, board texture, blockers, and action history so that the policy differentiates UTG/MP/CO/BTN/SB/BB automatically.
- At evaluation or tournament time, load the same profile for every seat.

## Key Files

- `poker_cfr/env/game.py`: 2-6 player NLHE engine (blinds, abstract bet sizes, side effects like runouts and showdowns).
- `poker_cfr/env/hand_eval.py`: 7-card evaluator plus hand bucket helpers.
- `poker_cfr/cfr/infoset.py`: Utilities for building infoset keys from game state observations.
- `poker_cfr/cfr/trainer.py`: CFR + MCCFR implementation with regret-matching nodes and average strategy extraction.
- `poker_cfr/cfr/storage.py`: Pickle save/load helpers for strategy tables.
- `poker_cfr/bot/agent.py`: Runtime agent that queries the stored strategy and applies optional adjustments.
- `poker_cfr/bot/opponent_model.py`: Lightweight opponent tendency tracker used by the agent.
- `poker_cfr/scripts/train_strategy.py`: CLI driver for training (takes stacks/blinds/iterations/algo/player-count).
- `poker_cfr/scripts/evaluate_strategy.py`: Heads-up evaluation loop.
- `poker_cfr/scripts/evaluate_strategy_multi.py`: Multiplayer evaluation loop.
- `poker_cfr/scripts/compare_iterations.py`: Trains/loads multiple strategies and plays a round-robin to measure how iteration counts or algorithms affect performance.
- `poker_cfr/scripts/inspect_strategy.py`: Dumps sample infosets and strategy probabilities.

## Training Options

- `--iterations`: Traversals per run. More iterations produce smoother strategies but take longer.
- `--num-players`: Seats in the game (2-6). The trainer updates whatever player is acting, so one run covers all seats.
- `--algo`: `cfr` (full enumeration) or `mccfr` (external sampling, preferred for 4-6 players).
- `--stack`, `--small-blind`, `--big-blind`: Game configuration.
- `--output`: Path to save the average-strategy pickle.
- `--cores`: Number of CPU cores to use for parallel training (default: 1 = sequential). Use this to speed up training on multi-core systems. For example, use `--cores 16` on your AMD Ryzen 6900, or `--cores 92` on EC2 instances.

## Next Steps

1. Align your environment observation to the infoset helpers so runtime and training abstractions match.
2. Tune the betting abstraction if you need different raise sizes or side pots.
3. Scale up training (50k-100k+ iterations) before deploying to a tournament environment.
4. Point your tournament bot (or any client) to the saved `strategy.pkl` and call `CFRAgent.act()` during play.
5. (Optional) Turn on the runtime resolver by instantiating `CFRAgent(enable_resolver=True)`
   and passing a live `GameState` into the observation to mix blueprint play with
   short real-time CFR solves.

Refer to `POKER_CFR_DESIGN_NOTES.md` for deeper implementation notes and ideas for extensions.
