import chess
import pickle
import numpy as np
from datetime import datetime


def get_date():
    return datetime.today().strftime('%Y-%m-%d--%H-%M-%S')


def save(games, filename=None):
    if filename is None:
        filename = "./datasets/" + get_date()
    with open(filename, 'wb') as f:
        pickle.dump(games, f)


def read(filename):
    games = []
    with open(filename, 'rb') as fp:
        games = pickle.load(fp)
    return games


def to_bitboard(board, pack=False):
    def bitboards_to_array(bb: np.ndarray) -> np.ndarray:
        bb = np.asarray(bb, dtype=np.uint64)[:, np.newaxis]
        s = 8 * np.arange(7, -1, -1, dtype=np.uint64)
        b = (bb >> s).astype(np.uint8)
        b = np.unpackbits(b, bitorder="little")
        return b

    black, white = board.occupied_co

    bitboards = np.array([
        black & board.pawns,
        black & board.knights,
        black & board.bishops,
        black & board.rooks,
        black & board.queens,
        black & board.kings,
        white & board.pawns,
        white & board.knights,
        white & board.bishops,
        white & board.rooks,
        white & board.queens,
        white & board.kings,
    ], dtype=np.uint64)

    rv = bitboards_to_array(bitboards)
    if pack:
        return np.packbits(rv)
    return rv


ord_a = ord('a')
ord_1 = ord('1')


def move_to_bitmask(move: chess.Move):
    rv = np.zeros((32,), dtype=np.float64)
    move = str(move)
    rv[ord(move[0]) - ord_a] = 1
    rv[ord(move[1]) - ord_1 + 8] = 1
    rv[ord(move[2]) - ord_a + 16] = 1
    rv[ord(move[3]) - ord_1 + 24] = 1
    # rv = np.zeros((128,), dtype=np.uint8)
    # rv[56 + move.from_square % 8 - 8 * (move.from_square // 8)] = 1
    # rv[56 + move.to_square % 8 - 8 * (move.to_square // 8) + 64] = 1
    return rv


def bitmask_to_move(bitmask):
    rv = ''
    rv += chr(ord_a + bitmask[:8].argmax())
    rv += chr(ord_1 + bitmask[8:16].argmax())
    rv += chr(ord_a + bitmask[16:24].argmax())
    rv += chr(ord_1 + bitmask[24:32].argmax())
    return rv
