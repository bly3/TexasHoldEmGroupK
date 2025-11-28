"""
Infoset encoding utilities.

Use the abstractions described in POKER_CFR_DESIGN_NOTES.md:
- position (IP/OOP or SB/BB)
- street
- SPR bucket
- hand strength bucket
- board texture
- blocker features
- abstract betting history
"""

from collections import Counter
from typing import List

from poker_cfr.env.hand_eval import hand_bucket as eval_hand_bucket

RANK_VALUE = {r: i + 2 for i, r in enumerate("23456789TJQKA")}


def spr_bucket(stack: int, pot: int) -> str:
    spr = stack / max(pot, 1)
    if spr < 2:
        return "SPR_L"
    if spr < 5:
        return "SPR_M"
    return "SPR_H"


def board_texture(board: List[str]) -> str:
    """
    Compress board texture into a tiny label.
    """
    if not board:
        return "BT_none"
    suits = [c[-1] for c in board]
    unique_suits = len(set(suits))
    paired = len(board) != len(set(c[:-1] for c in board))
    base = "rainbow"
    if unique_suits == 1:
        base = "mono"
    elif unique_suits == 2:
        base = "two"
    texture = "paired" if paired else "unpaired"
    return f"BT_{base}_{texture}"


def compute_blocker_features(hole: List[str], board: List[str]) -> str:
    """
    Returns a compact string like 'NF1_F1_S0'.
    """
    has_nut_flush_blocker = False
    has_flush_blocker = False
    has_straight_blocker = False

    if board:
        suits = [c[-1] for c in board]
        suit_counts = Counter(suits)
        flush_suit = next((s for s, cnt in suit_counts.items() if cnt >= 3), None)
        if flush_suit:
            for card in hole:
                if card[-1] == flush_suit:
                    has_flush_blocker = True
                    if card[0] == "A":
                        has_nut_flush_blocker = True

        # Straight blocker: if board has a 3+ card straight region, owning endpoints counts.
        board_ranks = sorted({RANK_VALUE[c[0]] for c in board})
        for start in range(2, 11):  # 5-card windows starting at 2..10
            window = set(range(start, start + 5))
            gap = window - set(board_ranks)
            if 0 < len(gap) <= 2:
                for card in hole:
                    if RANK_VALUE[card[0]] in gap:
                        has_straight_blocker = True
                        break
                if has_straight_blocker:
                    break

    return f"NF{int(has_nut_flush_blocker)}_F{int(has_flush_blocker)}_S{int(has_straight_blocker)}"


def encode_betting_history(history: List[str]) -> str:
    """
    Compress abstract actions into a small string.
    """
    return "_".join(history) if history else "start"


def encode_infoset(
    position: str,
    street: str,
    stack: int,
    pot: int,
    hand_bucket: str,
    board: List[str],
    blockers: str,
    history: List[str],
) -> str:
    """
    Build the infoset key. Ensure training and runtime use the same formatter.
    """
    return "|".join(
        [
            position,
            street,
            spr_bucket(stack, pot),
            hand_bucket,
            board_texture(board),
            blockers,
            f"H:{encode_betting_history(history)}",
        ]
    )


def compute_hand_bucket(hole: List[str], board: List[str]) -> str:
    return eval_hand_bucket(hole, board)
