# state_encoding.py
from typing import List, Dict, Any
from RL1010.src.RL1010.config import BOARD_SIZE, PIECES_IN_HAND, SCORE_SCALE, MOVE_SCALE

# Constants (tweak SCALEs later if you want)

STATE_DIM = BOARD_SIZE * BOARD_SIZE + PIECES_IN_HAND * MASK_SIZE * MASK_SIZE + 2
# 100 + 3*25 + 2 = 177


def piece_to_5x5_mask(piece: Dict[str, Any]) -> List[int]:
    """
    Convert a piece's 2D shape into a flattened 5x5 binary mask (length 25).

    The original shape in pieces.py is a small 2D list of 0/1.
    We copy it into the top-left of a 5x5 grid and pad with zeros.
    """
    shape = piece["shape"]          # e.g. [[1,1,1], [0,1,0], ...]
    h = len(shape)

    mask_flat = [0] * (MASK_SIZE * MASK_SIZE)

    for r in range(min(h, MASK_SIZE)):
        row = shape[r]
        w = len(row)
        for c in range(min(w, MASK_SIZE)):
            if row[c]:
                mask_flat[r * MASK_SIZE + c] = 1

    return mask_flat  # length 25


def encode_state(
    board: List[List[int]],
    hand: List[Dict[str, Any]],
    score: int,
    move_count: int,
) -> List[float]:
    """
    Encode the full game state as a flat vector:

      - board: 10x10 -> 100 values (0/1)
      - hand: up to 3 pieces, each as 5x5 -> 3*25 = 75 values (0/1)
      - extras: [score_norm, move_count_norm] -> 2 floats

    Total length: 177.
    """
    # --- Board (100) ---
    if len(board) != BOARD_SIZE or any(len(row) != BOARD_SIZE for row in board):
        raise ValueError(f"Board must be {BOARD_SIZE}x{BOARD_SIZE}")

    board_flat: List[float] = [float(cell) for row in board for cell in row]

    # --- Hand (75) ---
    hand_flat: List[float] = []
    for slot in range(PIECES_IN_HAND):
        if slot < len(hand) and hand[slot] is not None:
            mask25 = piece_to_5x5_mask(hand[slot])     # length 25
            hand_flat.extend(float(v) for v in mask25)
        else:
            # Empty slot -> all zeros
            hand_flat.extend([0.0] * (MASK_SIZE * MASK_SIZE))

    # --- Extras (2) ---
    score_norm = float(score) / SCORE_SCALE
    moves_norm = float(move_count) / MOVE_SCALE
    extras = [score_norm, moves_norm]

    state_vec = board_flat + hand_flat + extras

    # Optional sanity check
    assert len(state_vec) == STATE_DIM, f"Expected {STATE_DIM}, got {len(state_vec)}"

    return state_vec
