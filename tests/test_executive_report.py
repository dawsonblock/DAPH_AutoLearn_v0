"""Tests for DAPH v0.4 executive report generation."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from daph_learning.executive import (
    ActionDescriptor,
    ActionSpace,
    ExecutiveTaskRecord,
    ExecutiveQualificationResult,
    evaluate_qualification,
    generate_markdown_report,
    generate_json_report,
    write_report,
    binary_action_space,
)


def _make_records(n=20, n_groups=4):
    import numpy as np
    space = binary_action_space()
    sym_id = "action.symbolic_arithmetic"
    llm_id = "action.llm_direct"
    records = []
    rng = np.random.RandomState(42)
    for i in range(n):
        sym_u = rng.uniform(0.3, 0.9)
        llm_u = rng.uniform(0.2, 0.7)
        oracle_a = sym_id if sym_u >= llm_u else llm_id
        records.append(ExecutiveTaskRecord(
            task_id=f"t{i}",
            group_id=f"g{i % n_groups}",
            subtype="A" if i % 2 == 0 else "B",
            split="test",
            utilities={sym_id: sym_u, llm_id: llm_u},
            probabilities={sym_id: 0.6, llm_id: 0.4},
            selected_action=sym_id,
            oracle_action=oracle_a,
            p1_realized_utility=sym_u,
            p0_realized_utility=llm_u,
            oracle_utility=max(sym_u, llm_u),
            regret=max(sym_u, llm_u) - sym_u,
        ))
    return records


class TestMarkdownReport:
    def test_basic_report(self):
        records = _make_records()
        space = binary_action_space()
        result = evaluate_qualification(
            records, space, experiment_id="test", bootstrap_iterations=100)
        md = generate_markdown_report(result)
        assert "Executive Qualification Report" in md
        assert "test" in md
        assert "P1 - P0" in md
        assert "Utility Summary" in md

    def test_per_action_table(self):
        records = _make_records()
        space = binary_action_space()
        result = evaluate_qualification(
            records, space, experiment_id="test", bootstrap_iterations=100)
        md = generate_markdown_report(result, include_per_action=True)
        assert "action.symbolic_arithmetic" in md
        assert "action.llm_direct" in md

    def test_per_group_table(self):
        records = _make_records()
        space = binary_action_space()
        result = evaluate_qualification(
            records, space, experiment_id="test", bootstrap_iterations=100)
        md = generate_markdown_report(
            result, include_per_group=True, records=records)
        assert "Per-Group" in md
        assert "g0" in md

    def test_three_action_report(self):
        space = ActionSpace(actions=(
            ActionDescriptor(action_id="reasoning.direct"),
            ActionDescriptor(action_id="retrieval.vector"),
            ActionDescriptor(action_id="reasoning.decompose"),
        ))
        import numpy as np
        rng = np.random.RandomState(42)
        records = []
        for i in range(20):
            utils = {
                "reasoning.direct": rng.uniform(0.3, 0.9),
                "retrieval.vector": rng.uniform(0.2, 0.7),
                "reasoning.decompose": rng.uniform(0.1, 0.6),
            }
            sel = "reasoning.direct"
            oracle_a = max(utils, key=utils.get)
            records.append(ExecutiveTaskRecord(
                task_id=f"t{i}", group_id=f"g{i%4}",
                subtype="A", split="test",
                utilities=utils,
                probabilities={"reasoning.direct": 0.5, "retrieval.vector": 0.3, "reasoning.decompose": 0.2},
                selected_action=sel,
                oracle_action=oracle_a,
                p1_realized_utility=utils[sel],
                p0_realized_utility=utils["retrieval.vector"],
                oracle_utility=utils[oracle_a],
                regret=utils[oracle_a] - utils[sel],
            ))
        result = evaluate_qualification(
            records, space, experiment_id="three_action", bootstrap_iterations=100)
        md = generate_markdown_report(result)
        assert "reasoning.direct" in md
        assert "retrieval.vector" in md
        assert "reasoning.decompose" in md


class TestJsonReport:
    def test_basic_json(self):
        records = _make_records()
        space = binary_action_space()
        result = evaluate_qualification(
            records, space, experiment_id="test", bootstrap_iterations=100)
        d = generate_json_report(result, records=records)
        assert d["experiment_id"] == "test"
        assert "task_records" in d
        assert len(d["task_records"]) == 20
        assert "report_generated_at" in d

    def test_json_serializable(self):
        records = _make_records()
        space = binary_action_space()
        result = evaluate_qualification(
            records, space, experiment_id="test", bootstrap_iterations=100)
        d = generate_json_report(result)
        # Should not raise
        json.dumps(d)


class TestWriteReport:
    def test_write_to_dir(self, tmp_path):
        records = _make_records()
        space = binary_action_space()
        result = evaluate_qualification(
            records, space, experiment_id="write_test", bootstrap_iterations=100)
        paths = write_report(result, tmp_path, records=records)
        assert paths["markdown"].exists()
        assert paths["json"].exists()
        assert "write_test" in paths["markdown"].name
        assert "write_test" in paths["json"].name
        # Verify JSON is valid
        data = json.loads(paths["json"].read_text())
        assert data["experiment_id"] == "write_test"
