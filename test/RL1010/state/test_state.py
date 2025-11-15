# test_state_encoding.py
from RL1010.constants.state import BOARD_SIZE, PIECES_IN_HAND, MASK_SIZE, STATE_DIM
from RL1010.state.state import piece_to_5x5_mask, encode_state
import pytest

def make_board(fill_value=0):
    """Helper: create a BOARD_SIZE x BOARD_SIZE board."""
    return [[fill_value for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def test_piece_to_5x5_mask_single_block():
    piece = {"name": "single", "shape": [[1]]}
    mask = piece_to_5x5_mask(piece)

    assert isinstance(mask, list)
    assert len(mask) == MASK_SIZE * MASK_SIZE

    # Only one cell should be 1, at top-left
    assert sum(mask) == 1
    assert mask[0] == 1

    # Values should all be 0 or 1
    assert all(v in (0, 1) for v in mask)


def test_piece_to_5x5_mask_rectangular_shape():
    # 2x3 shape
    shape = [
        [1, 1, 0],
        [0, 1, 1],
    ]
    piece = {"name": "weird_rect", "shape": shape}
    mask = piece_to_5x5_mask(piece)

    assert len(mask) == MASK_SIZE * MASK_SIZE

    # Check top-left 2x3 region matches original shape
    for r in range(2):
        for c in range(3):
            assert mask[r * MASK_SIZE + c] == shape[r][c]

    # Check that everything below original shape rows is zero
    for r in range(2, MASK_SIZE):
        for c in range(MASK_SIZE):
            assert mask[r * MASK_SIZE + c] == 0


def test_encode_state_length_and_structure_three_pieces():
    board = make_board(0)

    hand = [
        {"name": "single", "shape": [[1]]},
        {"name": "horizontal_two", "shape": [[1, 1]]},
        {"name": "square", "shape": [[1, 1], [1, 1]]},
    ]

    score = 500
    move_count = 42

    state = encode_state(board, hand, score, move_count)

    # Overall length
    assert len(state) == STATE_DIM

    # Board segment should be first BOARD_SIZE^2 entries
    board_segment = state[: BOARD_SIZE * BOARD_SIZE]
    assert len(board_segment) == BOARD_SIZE * BOARD_SIZE
    assert all(v == 0.0 for v in board_segment)

    # Hand segment length
    hand_segment = state[
        BOARD_SIZE * BOARD_SIZE :
        BOARD_SIZE * BOARD_SIZE + PIECES_IN_HAND * MASK_SIZE * MASK_SIZE
    ]
    assert len(hand_segment) == PIECES_IN_HAND * MASK_SIZE * MASK_SIZE

    # Extras at the end
    extras = state[-2:]
    assert pytest.approx(extras[0]) == score / 1000.0
    assert pytest.approx(extras[1]) == move_count / 200.0


def test_encode_state_empty_slots_are_zeroed():
    board = make_board(0)

    # Only one piece in hand; remaining slots should be zeros
    hand = [
        {"name": "single", "shape": [[1]]},
    ]

    state = encode_state(board, hand, score=0, move_count=0)

    board_end = BOARD_SIZE * BOARD_SIZE
    hand_start = board_end
    hand_end = hand_start + PIECES_IN_HAND * MASK_SIZE * MASK_SIZE

    hand_segment = state[hand_start:hand_end]

    # First slot (first 25 values) should contain exactly one "1"
    slot0 = hand_segment[: MASK_SIZE * MASK_SIZE]
    assert sum(slot0) == 1.0

    # Remaining slots should be all zeros
    slot1 = hand_segment[MASK_SIZE * MASK_SIZE : 2 * MASK_SIZE * MASK_SIZE]
    slot2 = hand_segment[2 * MASK_SIZE * MASK_SIZE : 3 * MASK_SIZE * MASK_SIZE]
    assert all(v == 0.0 for v in slot1)
    assert all(v == 0.0 for v in slot2)


def test_encode_state_invalid_board_raises():
    # 9x10 board instead of 10x10
    bad_board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE - 1)]
    hand = []

    with pytest.raises(ValueError):
        encode_state(bad_board, hand, score=0, move_count=0)


def test_encode_state_is_deterministic():
    board = make_board(0)
    hand = [
        {"name": "single", "shape": [[1]]},
        {"name": "horizontal_two", "shape": [[1, 1]]},
    ]
    score = 123
    move_count = 7

    state1 = encode_state(board, hand, score, move_count)
    state2 = encode_state(board, hand, score, move_count)

    assert state1 == state2
