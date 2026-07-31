from __future__ import annotations

import ast
import operator
import re
from typing import Any, Mapping


class SymbolicMathError(ValueError):
    pass


class UnsafeExpressionError(SymbolicMathError):
    pass


class UnsupportedOperationError(SymbolicMathError):
    pass


class ResourceLimitError(SymbolicMathError):
    pass


MAX_EXPR_CHARS = 256
MAX_AST_NODES = 128
MAX_AST_DEPTH = 32
MAX_INT_DIGITS = 64
MAX_RESULT_BITS = 4096
MAX_INTEGER_BITS = 4096
MAX_EXPONENT = 12

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _require_int(value: Any, name: str = "value", *, max_bits: int = MAX_INTEGER_BITS) -> int:
    # bool is a subclass of int; require the exact type.
    if type(value) is not int:
        raise UnsafeExpressionError(f"{name} must be an integer, got {type(value).__name__}")
    if value.bit_length() > max_bits:
        raise ResourceLimitError(f"{name} exceeds {max_bits} bits")
    return value


def _check_result(value: int) -> int:
    if type(value) is not int:
        raise SymbolicMathError("symbolic evaluator produced a non-integer result")
    if value.bit_length() > MAX_RESULT_BITS:
        raise ResourceLimitError(f"result exceeds {MAX_RESULT_BITS} bits")
    return value


def _depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    if not children:
        return 1
    return 1 + max(_depth(child) for child in children)


# Node types that are explicitly permitted in the integer-expression
# grammar. Anything else is rejected.
_PERMITTED_NODE_TYPES: frozenset[type] = frozenset({
    ast.Expression,
    ast.Constant,
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Mod,
    ast.FloorDiv,
})


def _validate_tree(
    tree: ast.AST,
    *,
    max_ast_nodes: int = MAX_AST_NODES,
    max_depth: int = MAX_AST_DEPTH,
) -> None:
    nodes = list(ast.walk(tree))
    if len(nodes) > max_ast_nodes:
        raise ResourceLimitError(f"expression exceeds {max_ast_nodes} AST nodes")
    if _depth(tree) > max_depth:
        raise ResourceLimitError(f"expression exceeds AST depth {max_depth}")
    # Explicitly reject every node type that is not in the permitted set.
    for node in nodes:
        if type(node) not in _PERMITTED_NODE_TYPES:
            raise UnsafeExpressionError(
                f"disallowed expression node: {type(node).__name__}")


def _eval_node(node: ast.AST, *, max_integer_bits: int = MAX_INTEGER_BITS) -> int:
    if isinstance(node, ast.Constant):
        # Reject floats, booleans, strings, and any non-int constant.
        if isinstance(node.value, bool) or type(node.value) is not int:
            raise UnsafeExpressionError(
                f"non-integer constant: {node.value!r}")
        return _require_int(node.value, "constant", max_bits=max_integer_bits)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _check_result(
            _ALLOWED_UNARYOPS[type(node.op)](
                _eval_node(node.operand, max_integer_bits=max_integer_bits))
        )

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_node(node.left, max_integer_bits=max_integer_bits)
        right = _eval_node(node.right, max_integer_bits=max_integer_bits)
        if isinstance(node.op, (ast.Mod, ast.FloorDiv)) and right == 0:
            raise ZeroDivisionError("integer division or modulo by zero")
        return _check_result(_ALLOWED_BINOPS[type(node.op)](left, right))

    raise UnsafeExpressionError(
        f"disallowed expression node: {type(node).__name__}"
    )


def safe_eval_int_expr(
    expr: str,
    *,
    max_ast_nodes: int = MAX_AST_NODES,
    max_depth: int = MAX_AST_DEPTH,
    max_integer_bits: int = MAX_INTEGER_BITS,
    max_exponent: int = MAX_EXPONENT,
) -> int:
    """Section 6: bounded AST evaluator for a tiny exact-integer grammar.

    Permitted nodes (explicitly enumerated):

    * integer ``Constant`` (no floats, no booleans, no strings);
    * ``UnaryOp`` with ``UAdd`` / ``USub``;
    * ``BinOp`` with ``Add`` / ``Sub`` / ``Mult`` / ``FloorDiv`` / ``Mod``.

    Deliberately unsupported (rejected): names, calls, attributes,
    subscripts, lists/tuples/sets/dicts, comprehensions, lambdas, imports,
    floating-point literals, true division (``/``), arbitrary
    exponentiation (``**``), bitwise operators, comparisons, boolean
    operators, conditional expressions, assignment expressions, and
    generator expressions.

    Resource limits: ``max_ast_nodes``, ``max_depth``, ``max_integer_bits``
    (per literal), and ``max_exponent`` (reserved for future Pow support;
    currently Pow is rejected outright).
    """
    if not isinstance(expr, str):
        raise TypeError("expr must be str")
    if len(expr) > MAX_EXPR_CHARS:
        raise ResourceLimitError(f"expression exceeds {MAX_EXPR_CHARS} characters")

    # Reject any forbidden characters early so that true division ("/"),
    # exponentiation ("**"), and bitwise operators never reach the parser.
    _ALLOWED_CHARS = set("0123456789+-*/%() \t")
    # Note: "/" alone is rejected below at the AST level (true division);
    # "//" is permitted. We keep "/" allowed at the char level so "//"
    # parses, then reject ast.Div at the node level.
    if any(c not in _ALLOWED_CHARS for c in expr):
        raise UnsafeExpressionError(
            f"expression contains disallowed characters: {expr!r}")

    tree = ast.parse(expr, mode="eval")
    _validate_tree(tree, max_ast_nodes=max_ast_nodes, max_depth=max_depth)
    return _eval_node(tree.body, max_integer_bits=max_integer_bits)


def execute_integer_arithmetic(inputs: Mapping[str, Any]) -> int:
    a = _require_int(inputs.get("a"), "a")
    b = _require_int(inputs.get("b"), "b")
    op = inputs.get("op")
    if op == "+":
        return _check_result(a + b)
    if op == "-":
        return _check_result(a - b)
    if op == "*":
        return _check_result(a * b)
    if op == "%":
        if b == 0:
            raise ZeroDivisionError("modulo by zero")
        return _check_result(a % b)
    if op == "//":
        if b == 0:
            raise ZeroDivisionError("floor division by zero")
        return _check_result(a // b)
    if op == "gcd":
        import math
        return _check_result(math.gcd(a, b))
    if op == "lcm":
        import math
        if a == 0 or b == 0:
            return 0
        return _check_result(a * b // math.gcd(a, b))
    raise UnsupportedOperationError(f"unsupported integer operation: {op!r}")


def execute_modular_multiplication(inputs: Mapping[str, Any]) -> int:
    a = _require_int(inputs.get("a"), "a")
    b = _require_int(inputs.get("b"), "b")
    modulus = inputs.get("modulus", inputs.get("mod", inputs.get("m", inputs.get("c"))))
    modulus = _require_int(modulus, "modulus")
    if modulus == 0:
        raise ZeroDivisionError("modulus cannot be zero")
    return _check_result((a * b) % modulus)



def modular_inputs_from_specification(specification: str) -> dict[str, int]:
    """Extract only the narrow `a * b mod c` textual form.

    This is a fallback for supported modular tasks that lack structured inputs;
    it is not a general natural-language parser.
    """
    match = re.search(
        r"\bCompute\s+\(?\s*(-?\d+)\s*\*\s*(-?\d+)\s*\)?\s*(?:mod|%)\s*(-?\d+)(?:\.|\s|$)",
        specification,
        flags=re.IGNORECASE,
    )
    if not match:
        raise SymbolicMathError("could not extract modular multiplication operands")
    return {
        "a": _require_int(int(match.group(1)), "a"),
        "b": _require_int(int(match.group(2)), "b"),
        "modulus": _require_int(int(match.group(3)), "modulus"),
    }

def expression_from_specification(specification: str) -> str:
    """Extract the arithmetic prefix immediately following ``Compute``.

    Parsing stops before alphabetic trailing instructions, a period, or end of
    input. The extracted text is still passed through the bounded AST evaluator;
    this function is only a narrow text boundary isolator, not an evaluator.
    """
    match = re.search(
        r"\bCompute\s+([0-9+\-*%/()\s]+?)(?=[a-zA-Z]|\.|$)",
        specification,
        flags=re.IGNORECASE,
    )
    if not match or not match.group(1).strip():
        raise SymbolicMathError("could not extract a safe arithmetic expression")
    return match.group(1).strip()


def execute_task(task: Mapping[str, Any]) -> int:
    caps = set(task.get("capability_ids", []))
    inputs = task.get("inputs") or {}

    if "modular_multiplication" in caps:
        return execute_modular_multiplication(inputs)

    if "integer_arithmetic" in caps and {"a", "b", "op"} <= set(inputs):
        return execute_integer_arithmetic(inputs)

    # Fallback only for explicitly arithmetic capabilities.
    if caps & {"integer_arithmetic", "modular_multiplication"}:
        expr = expression_from_specification(str(task.get("specification", "")))
        return safe_eval_int_expr(expr)

    raise UnsupportedOperationError(
        f"task capabilities are not supported by symbolic executor: {sorted(caps)}"
    )
