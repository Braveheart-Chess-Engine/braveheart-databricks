import pytest

from selfplay_analysis.records import Record, parse_record

AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


def test_parse_record_reads_fen_score_and_result() -> None:
    assert parse_record(f"{AFTER_E4} | 35 | 0.5") == Record(AFTER_E4, 35, 0.5)


def test_parse_record_reads_losses_and_wins() -> None:
    assert parse_record(f"{AFTER_E4} | -1210 | 0.0") == Record(AFTER_E4, -1210, 0.0)
    assert parse_record(f"{AFTER_E4} | 412 | 1.0").result == 1.0


@pytest.mark.parametrize(
    "line",
    [
        "",
        "no separators",
        f"{AFTER_E4} | 35",
        f"{AFTER_E4} | 3.5 | 0.5",
        f"{AFTER_E4} | 35 | 0.7",
        " | 35 | 0.5",
    ],
)
def test_malformed_records_are_refused(line: str) -> None:
    with pytest.raises(ValueError):
        parse_record(line)
