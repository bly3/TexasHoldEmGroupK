"""
CLI helper to spin up multiple CFR bots against the Go infrastructure server.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from typing import List, Optional, Sequence

from poker_cfr.bot.agent import CFRAgent
from poker_cfr.client import InfrastructureBotClient, InfrastructureConfig

DEFAULT_STRATEGY = "poker_cfr/data/strategy.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Connect CFR bots to the Go infrastructure server."
    )
    parser.add_argument("--server", default="ws://localhost:8080/ws", help="WebSocket endpoint")
    parser.add_argument("--api-key", default="dev", help="API key expected by the server")
    parser.add_argument("--table", default="table-1", help="Table identifier to join")
    parser.add_argument(
        "--start-key",
        default=None,
        help="Host/start key (only the first bot uses it if provided)",
    )
    parser.add_argument(
        "--min-raise",
        type=int,
        default=10,
        help="Minimum raise increment to send (defaults to server BB=10; increase if server is configured differently)",
    )
    parser.add_argument(
        "--strategy",
        dest="strategies",
        action="append",
        help=(
            "Strategy pickle path. Provide multiple --strategy values to assign different "
            "files per bot (in launch order). Defaults to poker_cfr/data/strategy.pkl."
        ),
    )
    parser.add_argument(
        "--players",
        nargs="+",
        default=None,
        help="Explicit player IDs. Overrides --num-bots / --player-prefix when set.",
    )
    parser.add_argument("--num-bots", type=int, default=2, help="Number of bot instances to run")
    parser.add_argument("--player-prefix", default="bot-", help="Prefix used when auto-generating IDs")
    parser.add_argument("--bot-name-prefix", default="CFRBot", help="Prefix for CFRAgent names")
    parser.add_argument(
        "--play-style",
        choices=["base", "aggressive", "nitty"],
        default="base",
        help="Agent style adjustment",
    )
    parser.add_argument(
        "--enable-resolver",
        action="store_true",
        help="Enable the real-time subgame resolver inside CFRAgent",
    )
    parser.add_argument(
        "--resolver-iterations",
        type=int,
        default=600,
        help="Iterations per resolver invocation",
    )
    parser.add_argument(
        "--resolver-algo",
        choices=["cfr", "mccfr"],
        default="mccfr",
        help="Resolver algorithm",
    )
    parser.add_argument(
        "--resolver-weight",
        type=float,
        default=0.55,
        help="Blend weight between blueprint and resolver strategy",
    )
    parser.add_argument(
        "--resolver-trigger",
        default="turn",
        help="Earliest street to trigger resolver (preflop/flop/turn/river)",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=2.0,
        help="Initial reconnect delay in seconds",
    )
    parser.add_argument(
        "--max-reconnect-delay",
        type=float,
        default=15.0,
        help="Maximum reconnect delay in seconds",
    )
    parser.add_argument(
        "--min-players",
        type=int,
        default=2,
        help="Minimum seated players required before the host auto-starts a hand",
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Disable automatic `start_hand` requests from the host bot",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Root logging level (DEBUG, INFO, WARNING, ...)",
    )
    return parser.parse_args()


async def run_async(args: argparse.Namespace) -> None:
    start_key = args.start_key if args.start_key is not None else os.getenv("START_KEY") or "supersecret"
    config = InfrastructureConfig(
        server_url=args.server,
        api_key=args.api_key,
        table_id=args.table,
        start_key=start_key,
        min_raise=args.min_raise,
        reconnect_delay=args.reconnect_delay,
        max_reconnect_delay=args.max_reconnect_delay,
        auto_start=not args.no_auto_start,
        min_players_to_start=args.min_players,
        bot_name_prefix=args.bot_name_prefix,
    )
    strategies = args.strategies or [DEFAULT_STRATEGY]
    player_ids = _resolve_player_ids(args.players, args.num_bots, args.player_prefix)

    async def launch_bot(player_id: str, is_host: bool, strategy_path: str) -> None:
        agent_name = f"{config.bot_name_prefix}-{player_id}"
        agent = CFRAgent(
            strategy_path=strategy_path,
            name=agent_name,
            play_style=args.play_style,
            enable_resolver=args.enable_resolver,
            resolver_iterations=args.resolver_iterations,
            resolver_algo=args.resolver_algo,
            resolver_blueprint_weight=args.resolver_weight,
            resolver_trigger_street=args.resolver_trigger,
        )
        client = InfrastructureBotClient(
            player_id=player_id,
            agent=agent,
            config=config,
            is_host=is_host and bool(config.start_key),
        )
        await client.run_forever()

    tasks = []
    for idx, pid in enumerate(player_ids):
        strategy_path = _strategy_for(idx, strategies)
        tasks.append(asyncio.create_task(launch_bot(pid, is_host=(idx == 0), strategy_path=strategy_path)))
    await asyncio.gather(*tasks)


def _resolve_player_ids(
    explicit: Optional[Sequence[str]], count: int, prefix: str
) -> List[str]:
    if explicit:
        return list(explicit)
    return [f"{prefix}{i+1}" for i in range(count)]


def _strategy_for(idx: int, strategies: Sequence[str]) -> str:
    if not strategies:
        return DEFAULT_STRATEGY
    if idx < len(strategies):
        return strategies[idx]
    # If fewer strategies than bots, reuse the last provided file.
    return strategies[-1]


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()


