"""
CFR and MCCFR trainer for the simplified heads-up Hold'em abstraction.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
import random
import tempfile
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from poker_cfr.cfr.infoset import (
    compute_blocker_features,
    compute_hand_bucket,
    encode_infoset,
)
from poker_cfr.cfr.storage import save_strategy
from poker_cfr.env.game import GameState, initial_state


@dataclass
class Node:
    infoset: str
    actions: List[str]
    regret_sum: List[float] = field(default_factory=list)
    strategy_sum: List[float] = field(default_factory=list)
    visit_count: int = 0

    def __post_init__(self) -> None:
        n = len(self.actions)
        if not self.regret_sum:
            self.regret_sum = [0.0] * n
        if not self.strategy_sum:
            self.strategy_sum = [0.0] * n

    def get_strategy(self, realization_weight: float) -> List[float]:
        self.visit_count += 1
        positive = [max(r, 0.0) for r in self.regret_sum]
        norm = sum(positive)
        strategy = (
            [p / norm for p in positive]
            if norm > 0
            else [1.0 / len(self.actions)] * len(self.actions)
        )
        for i, p in enumerate(strategy):
            self.strategy_sum[i] += realization_weight * p
        return strategy

    def get_average_strategy(self) -> List[float]:
        total = sum(self.strategy_sum)
        if total <= 0:
            return [1.0 / len(self.actions)] * len(self.actions)
        return [s / total for s in self.strategy_sum]


class CFRTrainer:
    def __init__(self) -> None:
        self.node_map: Dict[str, Node] = {}
        self.num_players: int | None = None

    # --- Full CFR ---------------------------------------------------------
    def cfr(self, state: GameState, reach_probs: List[float]) -> List[float]:
        """
        Full-tree CFR that supports 2-6 players. Returns the utility vector for all
        players and updates regrets for whoever is acting by using the product of
        opponents' reach probabilities.
        """
        self._ensure_num_players(state.num_players)

        if state.is_terminal():
            return self._terminal_utilities(state)

        current = state.to_act
        actions = state.legal_actions()

        if not actions or current not in state.active_players:
            return self._terminal_utilities(state) if state.is_terminal() else [0.0] * self.num_players

        infoset_key = self._encode_state(state, current)
        node = self.node_map.get(infoset_key)
        if node is None:
            node = Node(infoset_key, actions)
            self.node_map[infoset_key] = node

        strategy = node.get_strategy(reach_probs[current])

        action_utils: List[List[float]] = []
        node_util = [0.0 for _ in range(self.num_players)]

        for i, action in enumerate(actions):
            next_state = state.next_state(action)
            next_reach = list(reach_probs)
            next_reach[current] *= strategy[i]
            child_utils = self.cfr(next_state, next_reach)
            action_utils.append(child_utils)
            for p in range(self.num_players):
                node_util[p] += strategy[i] * child_utils[p]

        counterfactual_reach = self._counterfactual_reach(reach_probs, current)
        current_util = node_util[current]
        for i in range(len(actions)):
            regret = action_utils[i][current] - current_util
            node.regret_sum[i] += counterfactual_reach * regret

        return node_util

    # --- MCCFR (external sampling) ---------------------------------------
    def cfr_sample(self, state: GameState, reach_probs: List[float], traverser: int) -> float:
        """
        External-sampling MCCFR. During a sampled traversal we update regrets only
        for the traverser while sampling a single action for the other players.
        """
        self._ensure_num_players(state.num_players)

        if state.is_terminal():
            utils = self._terminal_utilities(state)
            return utils[traverser]

        current = state.to_act
        actions = state.legal_actions()
        if not actions or current not in state.active_players:
            return state.utility(traverser) if state.is_terminal() else 0.0

        infoset_key = self._encode_state(state, current)
        node = self.node_map.get(infoset_key)
        if node is None:
            node = Node(infoset_key, actions)
            self.node_map[infoset_key] = node

        if current == traverser:
            strategy = node.get_strategy(reach_probs[current])
            action_utils = [0.0 for _ in actions]
            node_util = 0.0
            for i, action in enumerate(actions):
                next_state = state.next_state(action)
                next_reach = list(reach_probs)
                next_reach[current] *= strategy[i]
                action_utils[i] = self.cfr_sample(next_state, next_reach, traverser)
                node_util += strategy[i] * action_utils[i]

            counterfactual_reach = self._counterfactual_reach(reach_probs, traverser)
            for i in range(len(actions)):
                regret = action_utils[i] - node_util
                node.regret_sum[i] += counterfactual_reach * regret
            return node_util

        strategy = node.get_strategy(reach_probs[current])
        sampled_idx = self._sample_index(strategy)
        sampled_action = actions[sampled_idx]
        sample_prob = max(strategy[sampled_idx], 1e-12)
        next_state = state.next_state(sampled_action)
        next_reach = list(reach_probs)
        next_reach[current] *= strategy[sampled_idx]
        return self.cfr_sample(next_state, next_reach, traverser) / sample_prob

    # --- Strategy aggregation --------------------------------------------
    def build_average_strategy(self) -> Dict[str, Dict[str, float]]:
        strategy: Dict[str, Dict[str, float]] = {}
        for key, node in self.node_map.items():
            avg = node.get_average_strategy()
            strategy[key] = {action: avg[i] for i, action in enumerate(node.actions)}
        return strategy

    def train(self, iterations: int, initial_state_factory, algo: str = "cfr", train_player: int = 0, num_cores: int = 1, 
              stack_size: int = None, blinds: Tuple[int, int] = None,
              checkpoint_every: int | None = None, checkpoint_dir: str | None = None,
              checkpoint_prefix: str = "strategy_ckpt") -> Dict[str, Dict[str, float]]:
        """
        Convenience wrapper to run training and return average strategy.
        
        In CFR, we update regrets for whoever is acting at each node during tree traversal.
        So all players improve together in one training run. The train_player parameter
        is retained for compatibility but does not gate which seats are updated.
        
        Args:
            iterations: Number of training iterations
            initial_state_factory: Function that creates a new initial game state
            algo: "cfr" or "mccfr"
            train_player: Player to train (for compatibility, doesn't restrict updates)
            num_cores: Number of CPU cores to use for parallel training (1 = sequential)
            stack_size: Stack size for parallel training (extracted from factory if not provided)
            blinds: Blinds tuple (small, big) for parallel training (extracted from factory if not provided)
            checkpoint_every: Save a strategy snapshot every N iterations (after merge for parallel)
            checkpoint_dir: Directory for checkpoint files
            checkpoint_prefix: Filename prefix for checkpoints
        """
        start_time = time.time()
        print(f"Starting training: {iterations} iterations, algorithm={algo}, cores={num_cores}")
        checkpoint_path = None
        if checkpoint_every and checkpoint_every > 0:
            ckpt_dir = Path(checkpoint_dir or "poker_cfr/data/checkpoints")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = ckpt_dir
        
        if num_cores <= 1:
            # Sequential training with progress updates
            if algo == "cfr":
                for i in range(iterations):
                    state = initial_state_factory()
                    reach = [1.0] * state.num_players
                    self.cfr(state, reach)
                    
                    # Log progress every 10% or every 1000 iterations, whichever is more frequent
                    if (i + 1) % max(1, min(iterations // 10, 1000)) == 0 or (i + 1) == iterations:
                        elapsed = time.time() - start_time
                        rate = (i + 1) / elapsed if elapsed > 0 else 0
                        remaining = (iterations - i - 1) / rate if rate > 0 else 0
                        eta = timedelta(seconds=int(remaining))
                        node_map_size = len(self.node_map)
                        print(f"Progress: {i + 1}/{iterations} ({100*(i+1)/iterations:.1f}%) | "
                              f"Rate: {rate:.1f} iter/s | ETA: {eta} | Nodes: {node_map_size:,}")
                        
                        # Prune rarely-visited nodes periodically to maintain performance
                        if node_map_size > 50000 and (i + 1) % max(1, iterations // 20) == 0:
                            # Adaptive threshold based on dictionary size
                            if node_map_size > 200000:
                                threshold = 15
                            elif node_map_size > 100000:
                                threshold = 12
                            else:
                                threshold = 10
                            self._prune_node_map(min_visit_count=threshold)
                    
                    if checkpoint_path and (i + 1) % checkpoint_every == 0:
                        self._checkpoint_strategy(checkpoint_path, checkpoint_prefix, i + 1)
            else:
                for i in range(iterations):
                    first_state = initial_state_factory()
                    num_players = first_state.num_players
                    # Traverse once per player so everybody collects external-sampling updates
                    for traverser in range(num_players):
                        sampled_state = first_state if traverser == 0 else initial_state_factory()
                        reach = [1.0] * sampled_state.num_players
                        self.cfr_sample(sampled_state, reach, traverser)
                    
                    # Log progress every 10% or every 1000 iterations
                    if (i + 1) % max(1, min(iterations // 10, 1000)) == 0 or (i + 1) == iterations:
                        elapsed = time.time() - start_time
                        rate = (i + 1) / elapsed if elapsed > 0 else 0
                        remaining = (iterations - i - 1) / rate if rate > 0 else 0
                        eta = timedelta(seconds=int(remaining))
                        node_map_size = len(self.node_map)
                        print(f"Progress: {i + 1}/{iterations} ({100*(i+1)/iterations:.1f}%) | "
                              f"Rate: {rate:.1f} iter/s | ETA: {eta} | Nodes: {node_map_size:,}")
                        
                        # Prune rarely-visited nodes periodically to maintain performance
                        if node_map_size > 50000 and (i + 1) % max(1, iterations // 20) == 0:
                            # Adaptive threshold based on dictionary size
                            if node_map_size > 200000:
                                threshold = 15
                            elif node_map_size > 100000:
                                threshold = 12
                            else:
                                threshold = 10
                            self._prune_node_map(min_visit_count=threshold)
                    
                    if checkpoint_path and (i + 1) % checkpoint_every == 0:
                        self._checkpoint_strategy(checkpoint_path, checkpoint_prefix, i + 1)
        else:
            # Parallel training
            self._train_parallel(iterations, initial_state_factory, algo, num_cores, stack_size, blinds, start_time,
                                 checkpoint_every=checkpoint_every, checkpoint_path=checkpoint_path,
                                 checkpoint_prefix=checkpoint_prefix)
        
        total_time = time.time() - start_time
        print(f"Training completed in {timedelta(seconds=int(total_time))} ({total_time:.1f}s)")
        return self.build_average_strategy()
    
    def _train_parallel(self, iterations: int, initial_state_factory: Callable, algo: str, num_cores: int,
                       stack_size: int = None, blinds: Tuple[int, int] = None, start_time: float = None,
                       checkpoint_every: int | None = None, checkpoint_path: Path | None = None,
                       checkpoint_prefix: str = "strategy_ckpt") -> None:
        """
        Run training in parallel across multiple CPU cores.
        Each worker processes a batch of iterations and returns its node_map.
        We merge all node_maps by aggregating regrets and strategy sums.
        """
        if start_time is None:
            start_time = time.time()
        
        # Extract parameters from the factory by calling it once if not provided
        if stack_size is None or blinds is None:
            sample_state = initial_state_factory()
            if stack_size is None:
                stack_size = sample_state.initial_stacks[0] if sample_state.initial_stacks else 40
            if blinds is None:
                # Infer blinds from round_bets - typically SB posts small_blind, BB posts big_blind
                # Check the initial betting to determine blinds
                # This is a heuristic - ideally blinds should be passed as parameters
                if len(sample_state.round_bets) >= 2:
                    # Usually first two players post blinds
                    small_blind = min(sample_state.round_bets[0], sample_state.round_bets[1])
                    big_blind = max(sample_state.round_bets[0], sample_state.round_bets[1])
                    blinds = (small_blind, big_blind) if small_blind > 0 else (1, 2)
                else:
                    blinds = (1, 2)  # Default
            num_players = sample_state.num_players
        else:
            # Still need num_players
            sample_state = initial_state_factory()
            num_players = sample_state.num_players
        
        # Determine batch size per worker
        iterations_per_worker = max(1, iterations // num_cores)
        remainder = iterations % num_cores
        
        # Create work batches with parameters
        batches = []
        batch_sizes = []
        for i in range(num_cores):
            batch_size = iterations_per_worker + (1 if i < remainder else 0)
            if batch_size > 0:
                batches.append((batch_size, algo, stack_size, blinds[0], blinds[1], num_players))
                batch_sizes.append(batch_size)
        
        if not batches:
            return
        
        total_batch_iterations = sum(batch_sizes)
        print(f"Distributed {total_batch_iterations} iterations across {len(batches)} workers")
        print(f"Batch sizes: {batch_sizes}")
        
        # Run parallel training with deferred merging to maximize CPU usage
        # Strategy: Collect all worker results first, then merge after all workers complete
        # This keeps all cores busy during computation phase
        with multiprocessing.Pool(processes=num_cores) as pool:
            # Submit all worker tasks
            async_tasks = []
            task_to_batch = {}  # Map task to batch info
            for idx, batch in enumerate(batches):
                async_task = pool.apply_async(_train_worker, batch)
                async_tasks.append(async_task)
                task_to_batch[async_task] = (idx, batch_sizes[idx])
            
            # Collect results as workers finish (but don't merge yet - just log progress)
            worker_results = []  # List of (idx, worker_node_map, worker_num_players, batch_iterations)
            completed = 0
            total_workers = len(async_tasks)
            compute_start_time = time.time()
            
            # Use a set to track which tasks we've processed
            processed_tasks = set()
            
            print("Waiting for workers to complete (all cores busy)...")
            while completed < total_workers:
                # Check each task to see if it's ready
                for async_task in async_tasks:
                    if async_task in processed_tasks:
                        continue
                        
                    if async_task.ready():
                        payload_type, payload_data, worker_num_players = async_task.get()
                        idx, batch_iterations = task_to_batch[async_task]
                        completed += 1
                        processed_tasks.add(async_task)
                        
                        # Store result for later merging
                        worker_results.append((idx, payload_data, worker_num_players, batch_iterations, payload_type))
                        
                        # Log progress when each worker completes (lightweight logging)
                        elapsed = time.time() - compute_start_time
                        progress_pct = 100 * completed / total_workers
                        compute_rate = sum(b[3] for b in worker_results) / elapsed if elapsed > 0 else 0
                        
                        print(f"Worker {idx+1}/{total_workers} completed ({batch_iterations} iterations) | "
                              f"Progress: {completed}/{total_workers} ({progress_pct:.1f}%) | "
                              f"Elapsed: {timedelta(seconds=int(elapsed))} | "
                              f"Compute rate: {compute_rate:.1f} iter/s")
                        
                        break  # Process one at a time
                else:
                    # No tasks ready, sleep briefly to avoid busy-waiting
                    time.sleep(0.01)  # Shorter sleep since we're not blocking on merges
            
            compute_time = time.time() - compute_start_time
            print(f"\nAll {total_workers} workers completed in {timedelta(seconds=int(compute_time))} "
                  f"({compute_time:.1f}s)")
            print(f"Average compute rate: {total_batch_iterations / compute_time:.1f} iter/s")
            print(f"\nStarting merge phase...")
            
            # Now merge all worker results (this happens sequentially, but all computation is done)
            merge_start_time = time.time()
            merged_count = 0
            merged_iterations = 0
            next_ckpt_at = checkpoint_every if checkpoint_every and checkpoint_every > 0 else None
            for idx, payload_data, worker_num_players, batch_iterations, payload_type in worker_results:
                self._ensure_num_players(worker_num_players)
                
                # Merge this worker's results
                merge_start = time.time()
                if payload_type == "file":
                    worker_node_map = _load_node_map_from_file(payload_data)
                else:
                    worker_node_map = payload_data
                self._merge_node_map(worker_node_map)
                merge_time = time.time() - merge_start
                node_map_size = len(self.node_map)
                
                merged_count += 1
                merged_iterations += batch_iterations
                merge_progress = 100 * merged_count / total_workers
                print(f"  Merged worker {idx+1}/{total_workers} ({merge_progress:.1f}%) in {merge_time:.2f}s | "
                      f"Node map: {node_map_size:,}")
                
                # Clear the worker's node_map from memory immediately after merging
                del worker_node_map

                # Optional checkpoint after enough merged iterations
                if next_ckpt_at is not None and merged_iterations >= next_ckpt_at:
                    self._checkpoint_strategy(checkpoint_path, checkpoint_prefix, merged_iterations)
                    # Move the goalpost for the next checkpoint
                    next_ckpt_at += checkpoint_every
                
                # Prune periodically during merge (every 4 workers to maintain performance)
                if merged_count % 4 == 0 and node_map_size > 50000:
                    # Adaptive threshold: more aggressive pruning as dictionary grows
                    if node_map_size > 200000:
                        threshold = 15
                    elif node_map_size > 100000:
                        threshold = 12
                    else:
                        threshold = 10
                    
                    prune_start = time.time()
                    pruned = self._prune_node_map(min_visit_count=threshold)
                    prune_time = time.time() - prune_start
                    if pruned > 0 and prune_time > 0.1:
                        print(f"  Pruning took {prune_time:.2f}s")
            
            merge_time_total = time.time() - merge_start_time
            print(f"\nMerge phase completed in {timedelta(seconds=int(merge_time_total))} ({merge_time_total:.1f}s)")
            
            # Final prune before finishing
            final_size_before = len(self.node_map)
            if final_size_before > 50000:
                print(f"Performing final pruning...")
                final_prune_start = time.time()
                self._prune_node_map(min_visit_count=10)
                final_prune_time = time.time() - final_prune_start
                final_size_after = len(self.node_map)
                print(f"Final prune completed in {final_prune_time:.2f}s")
                print(f"Final node map size: {final_size_after:,} (reduced from {final_size_before:,})")
            
            # Final statistics
            total_elapsed = time.time() - start_time
            avg_rate = total_batch_iterations / total_elapsed if total_elapsed > 0 else 0
            merge_pct = 100 * merge_time_total / total_elapsed if total_elapsed > 0 else 0
            
            print(f"\n{'='*70}")
            print(f"Training Summary:")
            print(f"  Total time: {timedelta(seconds=int(total_elapsed))} ({total_elapsed:.1f}s)")
            print(f"  Compute phase: {timedelta(seconds=int(compute_time))} ({compute_time:.1f}s)")
            print(f"  Merge phase: {timedelta(seconds=int(merge_time_total))} ({merge_time_total:.1f}s, {merge_pct:.1f}%)")
            print(f"  Average rate: {avg_rate:.1f} iter/s")
            print(f"  Final node map size: {len(self.node_map):,} nodes")
            print(f"{'='*70}")

    def _merge_node_map(self, other_node_map: Dict[str, Node]) -> None:
        """
        Merge another node_map into self.node_map by aggregating regrets and strategy sums.
        """
        for infoset_key, other_node in other_node_map.items():
            if infoset_key not in self.node_map:
                # Create new node with same structure
                self.node_map[infoset_key] = Node(
                    infoset_key,
                    other_node.actions,
                    regret_sum=list(other_node.regret_sum),
                    strategy_sum=list(other_node.strategy_sum),
                    visit_count=other_node.visit_count
                )
            else:
                # Merge existing node: aggregate regrets and strategy sums
                my_node = self.node_map[infoset_key]
                # Ensure actions match
                if my_node.actions != other_node.actions:
                    raise ValueError(f"Actions mismatch for infoset {infoset_key}")
                
                # Aggregate regrets (sum them)
                for i in range(len(my_node.regret_sum)):
                    my_node.regret_sum[i] += other_node.regret_sum[i]
                
                # Aggregate strategy sums (sum them)
                for i in range(len(my_node.strategy_sum)):
                    my_node.strategy_sum[i] += other_node.strategy_sum[i]
                
                # Aggregate visit counts
                my_node.visit_count += other_node.visit_count

    def _prune_node_map(self, min_visit_count: int = 10) -> int:
        """
        Remove nodes with low visit counts to free memory and improve performance.
        This doesn't hurt strategy quality since rarely-visited nodes have little impact.
        
        Args:
            min_visit_count: Minimum visit count to keep a node (default: 10)
            
        Returns:
            Number of nodes pruned
        """
        size_before = len(self.node_map)
        to_remove = [
            key for key, node in self.node_map.items()
            if node.visit_count < min_visit_count
        ]
        
        for key in to_remove:
            del self.node_map[key]
        
        size_after = len(self.node_map)
        pruned_count = size_before - size_after
        
        if pruned_count > 0:
            reduction_pct = 100 * pruned_count / size_before
            print(f"  Pruned {pruned_count:,} nodes ({reduction_pct:.1f}%) with <{min_visit_count} visits. "
                  f"Node map: {size_before:,} → {size_after:,}")
        
        return pruned_count

    # --- Helpers ----------------------------------------------------------
    def _encode_state(self, state: GameState, current: int) -> str:
        # Get position for multiplayer (handles 2-6 players)
        position = state.get_position(current)
        return encode_infoset(
            position=position,
            street=state.street,
            stack=state.stacks[current],
            pot=state.pot,
            hand_bucket=compute_hand_bucket(state.hole_cards[current], state.board),
            board=state.board,
            blockers=compute_blocker_features(state.hole_cards[current], state.board),
            history=state.history,
        )

    @staticmethod
    def _sample_index(probs: List[float]) -> int:
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return i
        return len(probs) - 1

    def _terminal_utilities(self, state: GameState) -> List[float]:
        return [state.utility(p) for p in range(self.num_players or state.num_players)]

    def _counterfactual_reach(self, reach_probs: List[float], player: int) -> float:
        prod = 1.0
        for idx, prob in enumerate(reach_probs):
            if idx == player:
                continue
            prod *= prob
        return prod

    def _ensure_num_players(self, n: int) -> None:
        if self.num_players is None:
            self.num_players = n
        elif self.num_players != n:
            raise ValueError(f"Inconsistent num_players: expected {self.num_players}, got {n}")

    def _checkpoint_strategy(self, checkpoint_dir: Path | None, prefix: str, iteration: int) -> None:
        """Persist a snapshot of the current average strategy for safety."""
        if checkpoint_dir is None:
            return
        strategy = self.build_average_strategy()
        path = checkpoint_dir / f"{prefix}_{iteration}iters.pkl"
        save_strategy(str(path), strategy)
        print(f"[CKPT] Saved checkpoint after {iteration} iterations -> {path}")


def _train_worker(batch_size: int, algo: str, stack_size: int, small_blind: int, big_blind: int, num_players: int) -> Tuple[str, str, int]:
    """
    Worker function for parallel training. Runs a batch of iterations, writes the
    resulting node_map to a temporary file, and returns metadata so the parent
    process can load and merge it later. This keeps inter-process transfers small.
    """
    worker_id = os.getpid()  # Get process ID for logging
    # Ensure each worker explores a different trajectory stream
    seed = int(time.time_ns()) ^ worker_id ^ int.from_bytes(os.urandom(2), "little")
    random.seed(seed)
    
    trainer = CFRTrainer()
    
    # Create state factory with provided parameters
    def state_factory():
        return initial_state(
            stack_size=stack_size,
            blinds=(small_blind, big_blind),
            num_players=num_players
        )
    
    worker_start = time.time()
    
    if algo == "cfr":
        for i in range(batch_size):
            state = state_factory()
            reach = [1.0] * state.num_players
            trainer.cfr(state, reach)
            
            # Log progress every 10% for long batches
            if batch_size > 100 and (i + 1) % max(1, batch_size // 10) == 0:
                elapsed = time.time() - worker_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  [Worker PID {worker_id}] {i+1}/{batch_size} iterations ({100*(i+1)/batch_size:.1f}%) | "
                      f"Rate: {rate:.1f} iter/s")
    else:
        total_traversals = batch_size * num_players
        traversals_done = 0
        
        for i in range(batch_size):
            first_state = state_factory()
            num_players_val = first_state.num_players
            # Traverse once per player so everybody collects external-sampling updates
            for traverser in range(num_players_val):
                sampled_state = first_state if traverser == 0 else state_factory()
                reach = [1.0] * sampled_state.num_players
                trainer.cfr_sample(sampled_state, reach, traverser)
                traversals_done += 1
            
            # Log progress every 10% for long batches
            if batch_size > 100 and (i + 1) % max(1, batch_size // 10) == 0:
                elapsed = time.time() - worker_start
                rate = traversals_done / elapsed if elapsed > 0 else 0
                print(f"  [Worker PID {worker_id}] {i+1}/{batch_size} iterations "
                      f"({traversals_done}/{total_traversals} traversals, {100*(i+1)/batch_size:.1f}%) | "
                      f"Rate: {rate:.1f} traversals/s")
    
    # Persist result to disk so Windows pipes don't need to transfer huge objects
    node_map_path = _dump_node_map_to_file(trainer.node_map)
    return ("file", node_map_path, trainer.num_players or 0)


def _dump_node_map_to_file(node_map: Dict[str, Node]) -> str:
    fd, path = tempfile.mkstemp(prefix="cfr_node_map_", suffix=".pkl")
    with os.fdopen(fd, "wb") as f:
        pickle.dump(node_map, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def _load_node_map_from_file(path: str) -> Dict[str, Node]:
    try:
        with open(path, "rb") as f:
            node_map = pickle.load(f)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return node_map
