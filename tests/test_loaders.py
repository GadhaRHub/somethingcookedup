from pathlib import Path

import pytest

from csvplot.loaders import file_dialog_filter, load_file

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "dRR_Ti_10nm.csv"


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_sample_file():
    ds = load_file(SAMPLE)
    assert list(ds.frame.columns) == ["t_ps", "dR_over_R", "dR_over_R_minus_bg"]
    assert ds.numeric_columns == ["t_ps", "dR_over_R", "dR_over_R_minus_bg"]
    assert len(ds.frame) == 12021


@pytest.mark.parametrize(
    "name,text",
    [
        ("semi.csv", "a;b\n1;2\n3;4\n"),
        ("tabs.tsv", "a\tb\n1\t2\n3\t4\n"),
        ("spaces.dat", "a   b\n1   2\n3   4\n"),
        ("comments.csv", "# measured 2026\n# units: ps\na,b\n1,2\n3,4\n"),
    ],
)
def test_delimiters_and_comments(tmp_path, name, text):
    ds = load_file(write(tmp_path, name, text))
    assert list(ds.frame.columns) == ["a", "b"]
    assert ds.frame["b"].tolist() == [2, 4]


def test_no_header(tmp_path):
    ds = load_file(write(tmp_path, "raw.csv", "1,2\n3,4\n"))
    assert list(ds.frame.columns) == ["col 1", "col 2"]
    assert len(ds.frame) == 2


def test_mostly_numeric_text_column(tmp_path):
    ds = load_file(write(tmp_path, "gaps.csv", "x,y\n1,2\n2,--\n3,5\n"))
    assert ds.numeric_columns == ["x", "y"]


def test_rejects_non_numeric(tmp_path):
    with pytest.raises(ValueError):
        load_file(write(tmp_path, "words.csv", "a,b\nfoo,bar\n"))


def test_dialog_filter_lists_csv():
    assert "*.csv" in file_dialog_filter()
