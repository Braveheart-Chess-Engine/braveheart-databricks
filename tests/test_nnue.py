import hashlib
import os
import struct
from pathlib import Path

import pytest

from selfplay_analysis import nnue

KINGS = "4k3/8/8/8/8/8/8/4K3 {} - - 0 1"
PAWN = "4k3/8/8/8/8/8/4P3/4K3 {} - - 0 1"


def synthetic(
    feature: dict[tuple[int, int], int], bias: dict[int, int], us: dict[int, int]
) -> bytes:
    """A 128-wide net with one output bucket and no king buckets (197,378 bytes): feature
    weights by (feature index, neuron), feature biases and the side to move's output
    weights by neuron, everything else zero."""
    values = [0] * (768 * 128 + 128 + 2 * 128 + 1)
    for (index, neuron), weight in feature.items():
        values[index * 128 + neuron] = weight
    for neuron, value in bias.items():
        values[768 * 128 + neuron] = value
    for neuron, weight in us.items():
        values[768 * 128 + 128 + neuron] = weight
    return struct.pack(f"<{len(values)}h", *values)


def test_the_output_scale_and_the_side_to_move() -> None:
    # One neuron saturated at 255 with output weight 64: 255 * 255 * 64 / 255 is 16,320,
    # times 400 over 255 * 64 is 400 centipawns for the side to move.
    net = nnue.load_net(synthetic({}, {0: 255}, {0: 64}))
    assert (net.hidden, net.buckets, net.king_buckets) == (128, 1, 1)
    assert nnue.evaluate(net, [KINGS.format("w"), KINGS.format("b")]).tolist() == [400, -400]


def test_a_feature_is_seen_from_its_own_side() -> None:
    # A white pawn (kind 0) on e2 (square 12) is feature 12 from white's view and 436 from
    # black's: only white's accumulator sees the weight, and only the side to move's
    # output weights are set. 100 * 100 * 64 / 255 truncates to 2,509, then to 61.
    net = nnue.load_net(synthetic({(12, 1): 100}, {}, {1: 64}))
    assert nnue.evaluate(net, [PAWN.format("w"), PAWN.format("b")]).tolist() == [61, 0]


def test_division_truncates_toward_zero_as_the_engines_does() -> None:
    # The same position with the output weight negated: -2,509.8 truncates to -2,509 and
    # -61.49 to -61, where floor division would give -2,510 and -62.
    net = nnue.load_net(synthetic({(12, 1): 100}, {}, {1: -64}))
    assert nnue.evaluate(net, [PAWN.format("w")]).tolist() == [-61]


def test_no_positions() -> None:
    assert nnue.evaluate(nnue.load_net(synthetic({}, {}, {})), []).tolist() == []


def test_a_wrong_size_or_checksum_is_refused() -> None:
    with pytest.raises(ValueError, match="size"):
        nnue.load_net(b"\0" * 1000)
    data = synthetic({}, {}, {})
    with pytest.raises(ValueError, match="checksum"):
        nnue.load_net(data, sha256="00" * 32)
    assert nnue.load_net(data, sha256=hashlib.sha256(data).hexdigest()).hidden == 128
    assert nnue.load_net(data + b"\0" * 62).hidden == 128  # padded to a multiple of 64


def test_a_download_is_checked_before_it_is_used(tmp_path: Path) -> None:
    data = synthetic({}, {0: 255}, {0: 64})
    path = tmp_path / "net.nnue"
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    assert nnue.download_net(url=path.as_uri(), sha256=digest) == data
    with pytest.raises(ValueError, match="checksum"):
        nnue.download_net(url=path.as_uri(), sha256="00" * 32)


def _real_net() -> bytes:
    default = Path(__file__).parent / "fixtures" / f"{nnue.NET_ID}.nnue"
    path = Path(os.environ.get("BRAVEHEART_NET", default))
    if not path.is_file():
        if os.environ.get("NET_TESTS") == "required":
            pytest.fail(f"no net at {path}")
        pytest.skip("the net file is not here (CI downloads it)")
    return path.read_bytes()


def test_the_port_matches_the_engine_on_every_golden_position() -> None:
    net = nnue.load_net(_real_net(), sha256=nnue.NET_SHA256)
    assert (net.hidden, net.buckets, net.king_buckets) == (512, 8, 4)
    assert nnue.check_golden(net) >= 400


def test_a_golden_mismatch_names_the_position() -> None:
    with pytest.raises(ValueError, match="differs from the engine"):
        nnue.check_golden(nnue.load_net(synthetic({}, {0: 255}, {0: 64})))
