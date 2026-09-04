from generation.prompts import _table_to_sentences


def test_table_rows_become_header_value_sentences():
    text = (
        "| Name | Value |\n"
        "|---|---|\n"
        "| A | 1 |\n"
        "| B | 2 |"
    )
    result = _table_to_sentences(text)
    assert "Name: A | Value: 1" in result
    assert "Name: B | Value: 2" in result


def test_repeated_separator_row_does_not_hang():
    """A malformed table with a second separator row must not loop forever."""
    text = (
        "| Name | Value |\n"
        "|---|---|\n"
        "| A | 1 |\n"
        "|---|---|\n"
        "| B | 2 |"
    )
    result = _table_to_sentences(text)
    assert "Name: A | Value: 1" in result
    assert "Name: B | Value: 2" in result
