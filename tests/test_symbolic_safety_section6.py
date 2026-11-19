"""Section 6 — symbolic execution safety: no Python eval/exec/compile in
symbolic paths, and the bounded AST evaluator enforces the integer grammar.

Required tests (Section 20.3):

* test_no_eval_in_symbolic_paths
* test_safe_integer_arithmetic
* test_rejects_calls
* test_rejects_names
* test_rejects_attributes
* test_rejects_large_exponent
* test_rejects_excessive_ast_depth
* test_rejects_divide_by_zero

Plus additional coverage for floats, true division, modulo, nested
arithmetic, oversized integer growth, unsupported operators, and
malicious expressions.
"""

from __future__ import annotations

import ast
import tokenize
from io import StringIO
from pathlib import Path

import pytest

from daph_learning.tools.symbolic_math import (
    ResourceLimitError,
    UnsafeExpressionError,
    safe_eval_int_expr,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Symbolic-execution source files that must never use eval/exec/compile.
_SYMBOLIC_PATHS = [
    "src/daph_learning/tools/symbolic_math.py",
    "src/daph_learning/execution/symbolic_executor.py",
    "src/daph_learning/execution/real_backends.py",
    "src/daph_learning/data/semantic_parser.py",
]


# ------------------------------------------------------------------
# Static scan: no eval/exec/compile in symbolic paths
# ------------------------------------------------------------------

def _iter_python_tokens(source: str):
    return tokenize.generate_tokens(StringIO(source).readline)


def test_no_eval_in_symbolic_paths():
    """A static scan of the symbolic package must fail if ``eval(``,
    ``exec(``, or ``compile(`` appears outside explicitly approved test
    fixtures. ``re.compile`` and torch ``model.eval()`` are allowed."""
    violations: list[str] = []
    for rel in _SYMBOLIC_PATHS:
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            # Bare eval()/exec()/compile() calls.
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id in {"eval", "exec", "compile"}:
                violations.append(f"{rel}:{node.lineno}: {node.func.id}()")
        # Token-level guard: catch eval/exec/compile used in any form,
        # but allow attribute access like model.eval() / re.compile().
        try:
            prev = None
            for tok in _iter_python_tokens(source):
                if tok.type == tokenize.NAME and tok.string in {
                        "eval", "exec", "compile"}:
                    # Allow method-call form: <name>.eval(
                    if prev is not None and prev.type == tokenize.OP \
                            and prev.string == ".":
                        pass
                    else:
                        violations.append(
                            f"{rel}:{tok.start[0]}: bare {tok.string}()")
                prev = tok
        except tokenize.TokenError:
            pass

    assert not violations, (
        "eval/exec/compile found in symbolic execution paths:\n"
        + "\n".join(violations))


# ------------------------------------------------------------------
# Grammar acceptance
# ------------------------------------------------------------------

def test_safe_integer_arithmetic():
    assert safe_eval_int_expr("42") == 42
    assert safe_eval_int_expr("-42") == -42
    assert safe_eval_int_expr("+42") == 42
    assert safe_eval_int_expr("8137 * 296") == 8137 * 296
    assert safe_eval_int_expr("2 * (3 + 4) - 5") == 9
    assert safe_eval_int_expr("100 // 7") == 14
    assert safe_eval_int_expr("100 % 7") == 2
    assert safe_eval_int_expr("-(3 + 4)") == -7


def test_valid_positive_and_negative_integers():
    assert safe_eval_int_expr("0") == 0
    assert safe_eval_int_expr("999999") == 999999
    assert safe_eval_int_expr("-999999") == -999999


def test_nested_arithmetic():
    assert safe_eval_int_expr("((1 + 2) * (3 + 4)) - (5 // 2)") == 19


def test_floor_division():
    assert safe_eval_int_expr("7 // 2") == 3
    assert safe_eval_int_expr("-7 // 2") == -4


def test_modulo():
    assert safe_eval_int_expr("7 % 3") == 1
    assert safe_eval_int_expr("-7 % 3") == 2


# ------------------------------------------------------------------
# Grammar rejection
# ------------------------------------------------------------------

def test_rejects_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        safe_eval_int_expr("1 // 0")
    with pytest.raises(ZeroDivisionError):
        safe_eval_int_expr("1 % 0")


def test_rejects_calls():
    with pytest.raises((UnsafeExpressionError, SyntaxError)):
        safe_eval_int_expr("abs(-1)")
    with pytest.raises((UnsafeExpressionError, SyntaxError)):
        safe_eval_int_expr("min(1, 2)")


def test_rejects_names():
    with pytest.raises(UnsafeExpressionError):
        safe_eval_int_expr("x + 1")
    with pytest.raises(UnsafeExpressionError):
        safe_eval_int_expr("True")


def test_rejects_attributes():
    with pytest.raises((UnsafeExpressionError, SyntaxError)):
        safe_eval_int_expr("a.b")


def test_rejects_large_exponent():
    # Pow (**) is not in the permitted node set at all.
    with pytest.raises(UnsafeExpressionError):
        safe_eval_int_expr("2 ** 3")
    with pytest.raises(UnsafeExpressionError):
        safe_eval_int_expr("2 ** 100")


def test_rejects_excessive_ast_depth():
    deep = "(" * 200 + "1" + ")" * 200
    with pytest.raises(ResourceLimitError):
        safe_eval_int_expr(deep)


def test_rejects_excessive_ast_nodes():
    wide = " + ".join(["1"] * 300)
    with pytest.raises(ResourceLimitError):
        safe_eval_int_expr(wide)


def test_rejects_oversized_integer_growth():
    # A literal that exceeds the integer bit limit.
    huge = "9" * 2000
    with pytest.raises(ResourceLimitError):
        safe_eval_int_expr(huge)


def test_rejects_true_division():
    with pytest.raises(UnsafeExpressionError):
        safe_eval_int_expr("7 / 2")


def test_rejects_floats():
    with pytest.raises(UnsafeExpressionError):
        safe_eval_int_expr("1.5")
    with pytest.raises(UnsafeExpressionError):
        safe_eval_int_expr("1.0 + 2")


def test_rejects_strings():
    with pytest.raises(UnsafeExpressionError):
        safe_eval_int_expr("'42'")


def test_rejects_booleans():
    with pytest.raises(UnsafeExpressionError):
        safe_eval_int_expr("True")
    with pytest.raises(UnsafeExpressionError):
        safe_eval_int_expr("False")


def test_rejects_bitwise_operators():
    for expr in ("1 & 2", "1 | 2", "1 ^ 2", "~1", "1 << 2", "1 >> 2"):
        with pytest.raises(UnsafeExpressionError):
            safe_eval_int_expr(expr)


def test_rejects_comparisons():
    with pytest.raises(UnsafeExpressionError):
        safe_eval_int_expr("1 < 2")


def test_rejects_malicious_expressions():
    malicious = [
        '__import__("os").system("echo pwned")',
        'open("/etc/passwd").read()',
        '(lambda: 1)()',
        '[x for x in range(1)][0]',
        '{1:2}[1]',
        '(1, 2)[0]',
    ]
    for expr in malicious:
        with pytest.raises((UnsafeExpressionError, SyntaxError)):
            safe_eval_int_expr(expr)


def test_unicode_normalization_not_required_for_digits():
    # Full-width digits are NOT normalized; they are rejected as
    # disallowed characters. Only ASCII digits are accepted.
    with pytest.raises(UnsafeExpressionError):
        safe_eval_int_expr("\uff11\uff12")  # full-width 1, 2


def test_max_integer_bits_parameter_is_honored():
    # A 100-bit literal is fine under the default 4096-bit limit but
    # rejected under a tiny limit.
    big = str(2 ** 100)
    assert safe_eval_int_expr(big) == 2 ** 100
    with pytest.raises(ResourceLimitError):
        safe_eval_int_expr(big, max_integer_bits=64)
