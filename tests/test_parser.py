import pytest
from pathlib import Path
from ingestion.parser import DoclingParser

FIXTURE = Path("tests/fixtures/sample.pdf")


@pytest.mark.integration
def test_parser_extracts_text_and_pages():
    parsed = DoclingParser().parse(FIXTURE)

    assert "SC-4471" in parsed.markdown
    assert "Escalation" in parsed.markdown

    pages = {b.page for b in parsed.blocks if b.page is not None}
    assert pages == {1, 2}


@pytest.mark.integration
def test_parser_reports_text_density_for_ocr_decision():
    parsed = DoclingParser().parse(FIXTURE)
    # native-text PDF — well above the OCR trigger threshold
    assert parsed.chars_per_page > 50


def test_default_converter_has_ocr_disabled():
    from ingestion.parser import _default_converter
    from docling.datamodel.base_models import InputFormat

    converter = _default_converter()
    pdf_options = converter.format_to_options[InputFormat.PDF]
    assert pdf_options.pipeline_options.do_ocr is False


def test_default_converter_uses_fast_table_mode():
    from ingestion.parser import _default_converter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import TableFormerMode

    converter = _default_converter()
    pdf_options = converter.format_to_options[InputFormat.PDF]
    assert pdf_options.pipeline_options.table_structure_options.mode == TableFormerMode.FAST


class FakeTableItem:
    def __init__(self, df):
        self._df = df

    def export_to_dataframe(self, doc):
        return self._df


def test_export_table_df_returns_the_dataframe():
    import pandas as pd
    from ingestion.parser import _export_table_df

    df = pd.DataFrame({"only": ["a", "b", "c"]})
    assert _export_table_df(FakeTableItem(df), doc=None) is df


def test_export_table_df_returns_none_on_failure():
    from ingestion.parser import _export_table_df

    class Broken:
        def export_to_dataframe(self, doc):
            raise RuntimeError("boom")

    assert _export_table_df(Broken(), doc=None) is None


def test_single_column_table_is_not_treated_as_a_genuine_table():
    import pandas as pd
    from ingestion.parser import _looks_like_a_table

    single_col = pd.DataFrame({"only": ["a", "b", "c"]})
    assert _looks_like_a_table(single_col) is False


def test_multi_column_table_is_treated_as_a_genuine_table():
    import pandas as pd
    from ingestion.parser import _looks_like_a_table

    two_col = pd.DataFrame({"Name": ["a"], "Value": [1]})
    assert _looks_like_a_table(two_col) is True


def test_missing_dataframe_defaults_to_trusting_doclings_classification():
    from ingestion.parser import _looks_like_a_table

    assert _looks_like_a_table(None) is True
