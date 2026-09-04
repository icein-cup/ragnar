from ui.static_files import file_url


def test_file_url_encodes_name_and_appends_page_fragment():
    assert file_url("http://localhost:8510", "contract v2 (final).pdf", 12) == (
        "http://localhost:8510/contract%20v2%20%28final%29.pdf#page=12"
    )


def test_file_url_omits_page_fragment_when_none():
    assert file_url("http://localhost:8510", "contract.pdf") == (
        "http://localhost:8510/contract.pdf"
    )


def test_file_url_keeps_spaces_quoted_but_inserts_fragment_after():
    assert file_url("http://localhost:8510", "my file.pdf", 3) == (
        "http://localhost:8510/my%20file.pdf#page=3"
    )