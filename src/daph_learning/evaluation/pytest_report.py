"""v0.3.10.3.2-alpha — structured pytest reporting (Section 5).

Parses JUnit XML output from ``pytest --junitxml=<path>`` to extract
test counts (collected, passed, failed, skipped, errors, xfail, xpass)
and computes a deterministic collection hash from sorted test node IDs.

This replaces terminal-string parsing with structured XML parsing.
"""

from __future__ import annotations

import hashlib
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PytestReport:
    """Section 5: structured pytest report."""
    collected: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    xfail: int = 0
    xpass: int = 0
    collected_test_count: int = 0
    pytest_collection_sha256: str = ""
    source_tree_sha256: str = ""
    release_version: str = ""
    created_at: float = 0.0
    node_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "collected": self.collected,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "xfail": self.xfail,
            "xpass": self.xpass,
            "collected_test_count": self.collected_test_count,
            "pytest_collection_sha256": self.pytest_collection_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "release_version": self.release_version,
            "created_at": self.created_at,
        }


def parse_junitxml(path: str | Path) -> PytestReport:
    """Section 5: parse a JUnit XML file produced by
    ``pytest --junitxml=<path>``.

    Extracts collected, passed, failed, skipped, errors, xfail, xpass
    counts and the full list of test node IDs.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"junitxml not found: {p}")

    tree = ET.parse(p)
    root = tree.getroot()

    report = PytestReport()

    # JUnit XML structure: <testsuite> with <testcase> children.
    # pytest's junitxml has tests, failures, skipped, errors attributes
    # on the root or testsuite elements.
    testsuites = root.findall(".//testsuite")
    if not testsuites:
        # Single testsuite as root.
        testsuites = [root]

    node_ids: list[str] = []
    for suite in testsuites:
        report.collected += int(suite.get("tests", 0))
        report.failed += int(suite.get("failures", 0))
        report.skipped += int(suite.get("skipped", 0))
        report.errors += int(suite.get("errors", 0))

        for tc in suite.findall("testcase"):
            classname = tc.get("classname", "")
            name = tc.get("name", "")
            node_id = f"{classname}::{name}"
            node_ids.append(node_id)

            # Check for skipped/xfail/xpass.
            skipped = tc.find("skipped")
            if skipped is not None:
                skiptype = skipped.get("type", "")
                if skiptype == "xfail":
                    report.xfail += 1
                else:
                    report.skipped += 1
            else:
                failure = tc.find("failure")
                error = tc.find("error")
                if failure is not None:
                    pass  # already counted in suite failures
                elif error is not None:
                    pass  # already counted in suite errors
                else:
                    report.passed += 1

    # If suite-level counts didn't capture everything, recompute from
    # testcases.
    if report.collected == 0 and node_ids:
        report.collected = len(node_ids)

    report.node_ids = node_ids
    report.collected_test_count = len(node_ids)
    report.pytest_collection_sha256 = compute_collection_hash(node_ids)
    return report


def compute_collection_hash(node_ids: list[str]) -> str:
    """Section 5: deterministic SHA-256 of sorted collected test node
    IDs.

    .. code-block:: python

        node_ids = sorted(collected_node_ids)
        collection_hash = sha256(
            "\\n".join(node_ids).encode()
        ).hexdigest()
    """
    sorted_ids = sorted(node_ids)
    text = "\n".join(sorted_ids)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_test_report(
    junitxml_path: str | Path,
    *,
    source_hash: str = "",
    release_version: str = "",
) -> PytestReport:
    """Section 5: build a complete test report from JUnit XML.

    Parameters
    ----------
    junitxml_path : path
        Path to the JUnit XML file.
    source_hash : str
        Current source tree SHA-256 (full 64 chars).
    release_version : str
        Current release version string.

    Returns
    -------
    PytestReport
        Complete report with all fields populated.
    """
    report = parse_junitxml(junitxml_path)
    report.source_tree_sha256 = source_hash
    report.release_version = release_version
    report.created_at = time.time()
    return report


__all__ = [
    "PytestReport",
    "build_test_report",
    "compute_collection_hash",
    "parse_junitxml",
]
