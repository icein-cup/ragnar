from eval.ingest_hybridqa import split_prose


def test_split_prose_no_overlap_matches_old_fixed_windows():
    text = "x" * 100
    assert split_prose(text, 30) == [text[i:i + 30] for i in range(0, 100, 30)]


def test_split_prose_overlap_shares_text_between_consecutive_pieces():
    text = "x" * 100
    pieces = split_prose(text, 30, overlap_chars=10)
    assert len(pieces) > 1
    for a, b in zip(pieces, pieces[1:]):
        assert a[-10:] == b[:10]


def test_split_prose_short_text_is_not_split():
    assert split_prose("short", 30, overlap_chars=10) == ["short"]
