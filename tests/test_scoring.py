from daph_learning.evaluation.scoring import parse_final_or_exact, parse_legacy_first_int


def test_final_parser():
    assert parse_final_or_exact("FINAL: -123") == -123
    assert parse_final_or_exact("-123") == -123
    assert parse_final_or_exact("First multiply 345 by 821. FINAL: 283245") == 283245
    assert parse_final_or_exact("First multiply 345 by 821. Answer 283245") is None


def test_final_answer_parser():
    """The canonical FINAL_ANSWER: format must be accepted."""
    assert parse_final_or_exact("FINAL_ANSWER: -123") == -123
    assert parse_final_or_exact("FINAL_ANSWER: 42") == 42
    assert parse_final_or_exact("Reasoning here.\nFINAL_ANSWER: 283245") == 283245
    assert parse_final_or_exact("FINAL_ANSWER: 0") == 0


def test_both_formats_accepted():
    """Both FINAL: and FINAL_ANSWER: must parse correctly."""
    assert parse_final_or_exact("FINAL: 42") == 42
    assert parse_final_or_exact("FINAL_ANSWER: 42") == 42
    # When both appear, the first match wins (regex .search() behavior)
    result = parse_final_or_exact("FINAL_ANSWER: 1\nFINAL: 2")
    assert result == 1


def test_legacy_parser_documents_old_behavior():
    assert parse_legacy_first_int("First multiply 345 by 821. Answer 283245") == 345
