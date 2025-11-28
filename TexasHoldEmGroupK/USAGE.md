# How to Use Your Poker Bot

## Quick Start

### 1. Basic Usage - Import and Use

```python
from poker_cfr.bot.agent import CFRAgent

# Create a bot instance (loads the strategy)
bot = CFRAgent(
    strategy_path="poker_cfr/data/strategy.pkl",
    name="MyBot",
    enable_resolver=False,  # set True to run real-time subgame solves
)

# When it's your turn, prepare the observation
observation = {
    "position": "IP",           # "IP" (in position) or "OOP" (out of position)
    "street": "preflop",        # "preflop", "flop", "turn", or "river"
    "stack": 40,                # Your current stack size
    "pot": 3,                   # Current pot size
    "hole_cards": ["As", "Kh"], # Your two hole cards (format: "rank"+"suit")
    "board": [],                # Community cards (empty preflop, 3 on flop, etc.)
    "history": ["call"],        # Betting history (list of actions so far)
}

legal_actions = ["fold", "call", "bet_pot", "bet_allin"]

# Get the bot's action
action = bot.act(observation, legal_actions)
print(f"Bot chooses: {action}")
```

### 2. Card Format

Cards are strings with format: `"rank" + "suit"`
- Ranks: `2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K, A`
- Suits: `s` (spades), `h` (hearts), `d` (diamonds), `c` (clubs)

Examples:
- `"As"` = Ace of spades
- `"Kh"` = King of hearts
- `"2c"` = Two of clubs
- `"Td"` = Ten of diamonds

---

## Common Use Cases

### Use Case 1: Integration with a Tournament System

```python
from poker_cfr.bot.agent import CFRAgent

class MyPokerBot:
    def __init__(self):
        self.agent = CFRAgent("poker_cfr/data/strategy.pkl", name="CFRBot")
    
    def get_action(self, game_state):
        """
        Convert tournament's game_state to our observation format.
        """
        observation = {
            "position": "IP" if game_state.my_position == "BB" else "OOP",
            "street": game_state.street.lower(),
            "stack": game_state.my_stack,
            "pot": game_state.pot_size,
            "hole_cards": game_state.my_cards,  # Should be list like ["As", "Kh"]
            "board": game_state.community_cards,  # List like ["2s", "5h", "9d"]
            "history": game_state.action_history,  # List like ["call", "bet_pot"]
        }
        
        return self.agent.act(observation, game_state.legal_actions)

# Usage
bot = MyPokerBot()
action = bot.get_action(current_game_state)
```

### Use Case 2: Play Against the Bot (Manual Testing)

```python
from poker_cfr.bot.agent import CFRAgent
from poker_cfr.env.game import initial_state

# Create bot
bot = CFRAgent(
    "poker_cfr/data/strategy.pkl",
    name="CFRBot",
    enable_resolver=True,
    resolver_iterations=800,
)

# Start a game
state = initial_state(stack_size=40, blinds=(1, 2))

while not state.is_terminal():
    current = state.to_act
    legal_actions = state.legal_actions()
    
    if current == 0:  # Bot's turn
        observation = {
            "position": "OOP",
            "street": state.street,
            "stack": state.stacks[0],
            "pot": state.pot,
            "hole_cards": state.hole_cards[0],
            "board": state.board,
            "history": state.history,
            "game_state": state,  # pass GameState to enable resolver
        }
        action = bot.act(observation, legal_actions)
        print(f"Bot chooses: {action}")
    else:  # Your turn (human input)
        print(f"Your turn! Legal actions: {legal_actions}")
        action = input("Your action: ")
    
    state = state.next_state(action)
    print(f"Stacks: {state.stacks}, Pot: {state.pot}\n")
```

### Use Case 3: Evaluate Bot Performance

```bash
# Run the built-in evaluation script
python -m poker_cfr.scripts.evaluate_strategy --games 1000
```

Or programmatically:

```python
from poker_cfr.scripts.evaluate_strategy import play_game
from poker_cfr.bot.agent import CFRAgent

bot1 = CFRAgent("poker_cfr/data/strategy.pkl", "Bot1")
bot2 = CFRAgent("poker_cfr/data/strategy.pkl", "Bot2")

# Play a single game
chip_change_0, chip_change_1, actions = play_game(bot1, bot2, stack_size=40, blinds=(1, 2))
print(f"Player 0: {chip_change_0}, Player 1: {chip_change_1}")
```

### Use Case 4: Compare Different Training Iterations

Use the arena script to train multiple strategies with different iteration counts (or load existing ones) and run a round-robin:

```bash
# Train 5k- and 100k-iteration strategies, then play 600 total games per matchup
python -m poker_cfr.scripts.compare_iterations \
  --iterations 5000 100000 \
  --games 600 \
  --stack 40 --small-blind 1 --big-blind 2
```

- Add an existing strategy with `--strategies label=path` (e.g., `baseline=poker_cfr/data/strategy.pkl`).
- Seat positions are swapped automatically so each bot plays both SB and BB.
- Temporary strategies are deleted after the run unless you pass `--keep-strategies`.
- The script prints per-match breakdowns plus a leaderboard sorted by average chips won per game.

---

## Observation Format Reference

The `observation` dictionary must have these keys:

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `position` | str | "IP" or "OOP" | `"IP"` |
| `street` | str | Current street | `"preflop"` |
| `stack` | int | Your stack size | `40` |
| `pot` | int | Current pot size | `3` |
| `hole_cards` | List[str] | Your two cards | `["As", "Kh"]` |
| `board` | List[str] | Community cards | `["2s", "5h", "9d"]` |
| `history` | List[str] | Action history | `["call", "bet_pot"]` |

**Optional keys** (auto-computed if missing):
- `hand_bucket` - Hand strength bucket (auto-computed)
- `blockers` - Blocker features (auto-computed)
- `game_state` - Pass a `GameState` object to enable real-time resolving
- `force_resolve` - Boolean override to trigger the resolver immediately

---

## Real-Time Resolver Mode

Set `enable_resolver=True` when instantiating `CFRAgent` to activate a short CFR/MCCFR
search every time the observation includes `game_state`. By default the agent starts
resolving on the turn, blends the resolver output with the cached blueprint strategy,
and returns the mixture. Tune these knobs:

- `resolver_iterations` – number of CFR/MCCFR iterations per decision (default 600)
- `resolver_algo` – `"mccfr"` or `"cfr"`
- `resolver_blueprint_weight` – 0.0 → use resolver only, 1.0 → rely on blueprint
- `resolver_trigger_street` – earliest street to run the resolver (`"turn"` default)

```python
agent = CFRAgent(
    enable_resolver=True,
    resolver_iterations=1000,
    resolver_blueprint_weight=0.4,
)
```

Make sure your environment can supply a `GameState` instance (or a compatible clone)
inside the observation so the resolver can traverse the remaining tree. You can also
force a resolve on high-leverage spots by setting `observation["force_resolve"] = True`.

---

## Action Format

The bot returns one of these actions:
- `"fold"` - Fold your hand
- `"call"` - Call/check (same action)
- `"bet_pot"` - Bet pot-sized bet
- `"bet_allin"` - Go all-in

---

## Example: Complete Integration

```python
"""
Example: Integrating CFR bot with a tournament system
"""

from poker_cfr.bot.agent import CFRAgent

class TournamentBot:
    def __init__(self):
        # Load the trained strategy once at startup
        self.agent = CFRAgent(
            strategy_path="poker_cfr/data/strategy.pkl",
            name="CFRBot"
        )
    
    def act(self, tournament_state):
        """
        Called by tournament system when it's your turn.
        
        Args:
            tournament_state: Your tournament system's game state object
            
        Returns:
            str: One of ["fold", "call", "bet_pot", "bet_allin"]
        """
        # Convert tournament state to our observation format
        observation = self._convert_to_observation(tournament_state)
        
        # Get legal actions from tournament
        legal_actions = tournament_state.get_legal_actions()
        
        # Get bot's decision
        action = self.agent.act(observation, legal_actions)
        
        return action
    
    def _convert_to_observation(self, state):
        """Helper to convert tournament state to our format."""
        return {
            "position": "IP" if state.am_in_position() else "OOP",
            "street": state.street.lower(),
            "stack": state.my_stack,
            "pot": state.pot,
            "hole_cards": [str(card) for card in state.my_cards],
            "board": [str(card) for card in state.board],
            "history": [str(action) for action in state.history],
        }

# Usage in tournament
bot = TournamentBot()

while tournament_running:
    if is_my_turn():
        action = bot.act(get_current_state())
        submit_action(action)
```

---

## Troubleshooting

### Bot returns random actions?
- Check that your observation format matches exactly
- Verify the strategy file exists: `poker_cfr/data/strategy.pkl`
- The bot falls back to random if it can't find the situation in its strategy

### Bot seems to play poorly?
- The strategy was trained for specific stack/blind settings (default: 40 chips, 1/2 blinds)
- For different settings, retrain with matching parameters:
  ```bash
  python -m poker_cfr.scripts.train_strategy --stack 100 --big-blind 5 --iterations 50000
  ```

### Card format errors?
- Make sure cards are strings: `"As"` not `"A♠"`
- Use the format: rank (2-9, T, J, Q, K, A) + suit (s, h, d, c)

---

## Next Steps

1. **Test the bot**: Use `evaluate_strategy.py` to see how it performs
2. **Integrate**: Adapt the observation format to your tournament system
3. **Tune**: Retrain with different parameters if needed
4. **Deploy**: Use in your tournament!

For more details, see `EXPLANATION.md` for how the bot works.

