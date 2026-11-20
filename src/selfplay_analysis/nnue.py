"""Braveheart's NNUE evaluation in NumPy: the net's raw static evaluation of a position,
with no search and no correction history.

A port of the engine's inference at `v0.54.0` (`crates/search/src/nnue.rs`, and the file
layout in the engine's `docs/nnue.md`): 768 piece-square inputs per perspective, in one of
four views chosen by that side's king square (mirrored to files a-d), a hidden layer per
perspective, SCReLU, and one output layer per band of piece counts. The arithmetic is the
engine's integer arithmetic, so the result is the engine's exactly, and `check_golden`
holds the port to positions the engine itself evaluated.
"""

from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: The net the engine embeds at v0.54.0, and where its file is public.
NET_ID = "net8x160-l050-r43390801430"
NET_SHA256 = "8ea67605f98048d78167e659387627396a256e54ecc1e9e94e9c5fbe04dd556c"
NET_URL = (
    "https://raw.githubusercontent.com/Braveheart-Chess/braveheart/v0.54.0"
    f"/crates/search/nets/{NET_ID}.nnue"
)
#: Positions with the engine's own evaluation of them (`fen<TAB>centipawns`, white's point
#: of view): 30 per output bucket for each side to move.
GOLDEN = Path(__file__).resolve().parent / "data" / "net8x160-golden.tsv"

INPUTS = 768
QA = 255  # hidden activations are clipped to 0..=QA
QB = 64  # the output weights' quantization
SCALE = 400  # centipawns per unit of the network's raw output
WIDTHS = (128, 256, 512, 768)
OUTPUT_BUCKETS = (1, 8)
KING_BUCKETS = (1, 4)
#: The king bucket of each square a mirrored king can stand on, from its own side: files
#: a-d of ranks 1 to 8. Part of the file format.
KING_BUCKET_LAYOUT = (0, 0, 1, 1, 2, 2, 2, 2) + (3,) * 24
_KINDS = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}
_BATCH = 2048


@dataclass(frozen=True, eq=False)
class Net:
    """A network's quantized weights, as int16 arrays: `feature_weights` is
    (768 * king_buckets, hidden), a row per feature; `output_us` and `output_them` are
    (buckets, hidden), the side to move's half and the other side's."""

    hidden: int
    king_buckets: int
    buckets: int
    feature_weights: np.ndarray
    feature_bias: np.ndarray
    output_us: np.ndarray
    output_them: np.ndarray
    output_bias: np.ndarray


def _file_bytes(hidden: int, buckets: int, king_buckets: int) -> int:
    return 2 * (INPUTS * king_buckets * hidden + hidden + buckets * (2 * hidden + 1))


def load_net(data: bytes, sha256: str | None = None) -> Net:
    """Read a network from its file bytes. The width, the output buckets and the king
    buckets are the ones whose size the file has (up to 63 bytes of padding are allowed);
    a file of any other size, or one whose SHA-256 is not `sha256` when given, raises
    ValueError."""
    if sha256 is not None:
        found = hashlib.sha256(data).hexdigest()
        if found != sha256:
            raise ValueError(f"the net's checksum is {found}, not {sha256}")
    for hidden in WIDTHS:
        for buckets in OUTPUT_BUCKETS:
            for king_buckets in KING_BUCKETS:
                size = _file_bytes(hidden, buckets, king_buckets)
                if size <= len(data) <= size + 63:
                    words = np.frombuffer(data, dtype="<i2", count=size // 2)
                    features = INPUTS * king_buckets * hidden
                    outputs = words[features + hidden : features + hidden + buckets * 2 * hidden]
                    outputs = outputs.reshape(buckets, 2, hidden)
                    return Net(
                        hidden=hidden,
                        king_buckets=king_buckets,
                        buckets=buckets,
                        feature_weights=words[:features].reshape(INPUTS * king_buckets, hidden),
                        feature_bias=words[features : features + hidden],
                        output_us=outputs[:, 0, :],
                        output_them=outputs[:, 1, :],
                        output_bias=words[size // 2 - buckets :],
                    )
    raise ValueError(f"a net of {len(data):,} bytes has no size the engine reads")


def download_net(url: str = NET_URL, sha256: str = NET_SHA256) -> bytes:
    """The net file's bytes from `url`, refused (ValueError) unless they are a net with
    that SHA-256."""
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    load_net(data, sha256=sha256)
    return data


def _view(net: Net, king: int) -> tuple[int, int]:
    """A perspective's view: the offset of its king bucket's features and the XOR that
    mirrors a square's file, from its own king's square seen from its own side."""
    if net.king_buckets == 1:
        return 0, 0
    file = king % 8
    flip, half = (7, 7 - file) if file > 3 else (0, file)
    return INPUTS * KING_BUCKET_LAYOUT[(king // 8) * 4 + half], flip


def _features(net: Net, fen: str) -> tuple[list[int], list[int], bool]:
    """The active feature indices from white's and black's perspective, and whether white
    is to move."""
    board, stm = fen.split(" ", 2)[:2]
    pieces = []
    kings = [0, 0]
    for row, rank in enumerate(board.split("/")):
        square = 8 * (7 - row)
        for char in rank:
            if char.isdigit():
                square += int(char)
                continue
            is_black = char.islower()
            kind = _KINDS[char.lower()]
            if kind == 5:
                kings[is_black] = square
            pieces.append((is_black, kind, square))
            square += 1
    white_base, white_flip = _view(net, kings[0])
    black_base, black_flip = _view(net, kings[1] ^ 56)
    white = [
        white_base + ((is_black * 384 + kind * 64 + sq) ^ white_flip)
        for is_black, kind, sq in pieces
    ]
    black = [
        black_base + (((not is_black) * 384 + kind * 64 + (sq ^ 56)) ^ black_flip)
        for is_black, kind, sq in pieces
    ]
    return white, black, stm == "w"


def _truncating(a: np.ndarray, b: int) -> np.ndarray:
    """a / b rounded toward zero, as Rust's integer division does (NumPy's floors)."""
    return np.where(a >= 0, a // b, -((-a) // b))


def evaluate(net: Net, fens: Sequence[str]) -> np.ndarray:
    """The net's static evaluation of each position, in centipawns from white's point of
    view (int32)."""
    out = np.zeros(len(fens), dtype=np.int32)
    # A zero row at the end is the index a position with fewer than 32 pieces pads with.
    weights = np.vstack([net.feature_weights, np.zeros((1, net.hidden), dtype=np.int16)])
    pad = len(weights) - 1
    band = -(-32 // net.buckets)
    bias = net.feature_bias.astype(np.int32)
    for start in range(0, len(fens), _BATCH):
        batch = fens[start : start + _BATCH]
        white = np.full((len(batch), 32), pad, dtype=np.int32)
        black = np.full((len(batch), 32), pad, dtype=np.int32)
        white_to_move = np.zeros(len(batch), dtype=bool)
        pieces = np.zeros(len(batch), dtype=np.int64)
        for row, fen in enumerate(batch):
            ours, theirs, white_to_move[row] = _features(net, fen)
            white[row, : len(ours)] = ours
            black[row, : len(theirs)] = theirs
            pieces[row] = len(ours)
        acc_white = bias + weights[white].sum(axis=1, dtype=np.int32)
        acc_black = bias + weights[black].sum(axis=1, dtype=np.int32)
        side = white_to_move[:, None]
        us = np.clip(np.where(side, acc_white, acc_black), 0, QA).astype(np.int64)
        them = np.clip(np.where(side, acc_black, acc_white), 0, QA).astype(np.int64)
        bucket = np.minimum(np.maximum(pieces - 2, 0) // band, net.buckets - 1)
        total = (us * us * net.output_us[bucket]).sum(axis=1) + (
            them * them * net.output_them[bucket]
        ).sum(axis=1)
        raw = _truncating(total, QA) + net.output_bias[bucket]
        mover = _truncating(raw * SCALE, QA * QB)
        out[start : start + len(batch)] = np.where(white_to_move, mover, -mover)
    return out


def check_golden(net: Net) -> int:
    """Evaluate the golden positions and return how many there are; a position whose
    evaluation is not the engine's raises ValueError naming it. A wrong net, a wrong
    layout or a wrong rounding all show here, before any real position is read."""
    fens, expected = [], []
    for line in GOLDEN.read_text().splitlines():
        fen, centipawns = line.split("\t")
        fens.append(fen)
        expected.append(int(centipawns))
    found = evaluate(net, fens)
    for fen, want, got in zip(fens, expected, found.tolist(), strict=True):
        if got != want:
            raise ValueError(f"{fen}: the port gives {got}, which differs from the engine's {want}")
    return len(fens)
