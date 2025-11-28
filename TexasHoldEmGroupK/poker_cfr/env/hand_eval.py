"""
Texas Hold'em hand evaluation utilities.
"""

from typing import List, Tuple


def evaluate_winner(hole_a: List[str], hole_b: List[str], board: List[str]) -> int:
    """
    Returns 0 if player A wins, 1 if player B wins, -1 for tie.
    """
    rank_a = _best_hand_score(hole_a + board)
    rank_b = _best_hand_score(hole_b + board)
    if rank_a > rank_b:
        return 0
    if rank_b > rank_a:
        return 1
    return -1


def evaluate_multi_winner(
    hole_cards_list: List[List[str]],
    player_indices: List[int],
    board: List[str]
) -> List[int]:
    """
    Returns list of winning player indices (can be multiple for ties).
    Evaluates all players' hands and returns those with the best hand.
    """
    if not hole_cards_list or not player_indices:
        return []
    
    if len(hole_cards_list) != len(player_indices):
        raise ValueError("hole_cards_list and player_indices must have same length")
    
    # Evaluate all hands
    hand_scores = []
    for hole in hole_cards_list:
        score = _best_hand_score(hole + board)
        hand_scores.append(score)
    
    # Find best score
    best_score = max(hand_scores)
    
    # Return all players with best score
    winners = [
        player_indices[i]
        for i, score in enumerate(hand_scores)
        if score == best_score
    ]
    
    return winners


def hand_bucket(hole: List[str], board: List[str]) -> str:
    """
    Rough hand strength bucket.
    Replace with equity approximation vs fixed ranges for better play.
    """
    score = _best_hand_score(hole + board)
    category = score[0]
    if category >= 7:
        return "HB4"  # straight flush / quads
    if category == 6:
        return "HB3"  # full house
    if category == 5 or category == 4:
        return "HB2"  # flush/straight
    if category == 3 or category == 2:
        return "HB1"  # trips/two pair
    return "HB0"  # pair/high card


# --- Internal helpers -----------------------------------------------------

RANKS = "23456789TJQKA"
RANK_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}  # 2..14


def _best_hand_score(cards: List[str]) -> Tuple:
    """
    Returns a sortable tuple representing the best 5-card hand from 7 cards.
    Higher tuple compares greater.
    """
    ranks = [RANK_VALUE[c[0]] for c in cards]
    suits = [c[1] for c in cards]
    # Count ranks
    count_by_rank = {}
    for r in ranks:
        count_by_rank[r] = count_by_rank.get(r, 0) + 1
    # Flush detection
    suit_counts = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1
    flush_suit = next((s for s, c in suit_counts.items() if c >= 5), None)
    flush_ranks = [r for r, s in sorted(zip(ranks, suits), reverse=True) if s == flush_suit] if flush_suit else []
    straight_high = _straight_high(ranks)
    straight_flush_high = _straight_high(flush_ranks) if flush_ranks else None

    # Straight flush
    if straight_flush_high is not None:
        return (8, straight_flush_high)

    # Four of a kind
    quads = [r for r, c in count_by_rank.items() if c == 4]
    if quads:
        quad = max(quads)
        kicker = max(r for r in ranks if r != quad)
        return (7, quad, kicker)

    # Full house
    trips = sorted([r for r, c in count_by_rank.items() if c == 3], reverse=True)
    pairs = sorted([r for r, c in count_by_rank.items() if c == 2], reverse=True)
    if trips and (len(trips) > 1 or pairs):
        trip_val = trips[0]
        pair_val = trips[1] if len(trips) > 1 else pairs[0]
        return (6, trip_val, pair_val)

    # Flush
    if flush_suit:
        top_five = flush_ranks[:5]
        return (5, *top_five)

    # Straight
    if straight_high is not None:
        return (4, straight_high)

    # Three of a kind
    if trips:
        trip_val = trips[0]
        kickers = sorted([r for r in ranks if r != trip_val], reverse=True)[:2]
        return (3, trip_val, *kickers)

    # Two pair
    if len(pairs) >= 2:
        high, low = pairs[0], pairs[1]
        kicker = max(r for r in ranks if r not in (high, low))
        return (2, high, low, kicker)

    # One pair
    if pairs:
        pair_val = pairs[0]
        kickers = sorted([r for r in ranks if r != pair_val], reverse=True)[:3]
        return (1, pair_val, *kickers)

    # High card
    top_five = sorted(ranks, reverse=True)[:5]
    return (0, *top_five)


def _straight_high(ranks: List[int]) -> int:
    """Return high card of best straight, or None."""
    if len(ranks) < 5:
        return None
    uniq = sorted(set(ranks))
    if len(uniq) < 5:
        return None
    # Handle wheel (A-2-3-4-5)
    if {14, 5, 4, 3, 2}.issubset(uniq):
        return 5
    # Sliding window from high to low
    for i in range(len(uniq) - 5, -1, -1):
        window = uniq[i : i + 5]
        if window[-1] - window[0] == 4:
            return window[-1]
    return None
