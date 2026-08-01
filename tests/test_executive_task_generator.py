"""Tests for the diverse task generator."""

from daph_learning.executive.task_generator import (
    generate_diverse_tasks,
    build_retrieval_store,
)


class TestGenerateDiverseTasks:
    def test_generates_correct_count(self):
        tasks = generate_diverse_tasks(n_tasks=80, n_groups=10, seed=42)
        assert len(tasks) == 80

    def test_has_8_subtypes(self):
        tasks = generate_diverse_tasks(n_tasks=80, n_groups=10, seed=42)
        subtypes = {t["subtype"] for t in tasks}
        assert len(subtypes) == 8
        assert "small_add" in subtypes
        assert "medium_add" in subtypes
        assert "large_add" in subtypes
        assert "small_mul" in subtypes
        assert "medium_mul" in subtypes
        assert "word_problem" in subtypes
        assert "multi_step" in subtypes
        assert "comparison" in subtypes

    def test_all_tasks_have_required_fields(self):
        tasks = generate_diverse_tasks(n_tasks=16, n_groups=4, seed=42)
        for t in tasks:
            assert "task_id" in t
            assert "prompt" in t
            assert "answer" in t
            assert "subtype" in t
            assert "group_id" in t

    def test_reproducible_with_same_seed(self):
        t1 = generate_diverse_tasks(n_tasks=16, n_groups=4, seed=42)
        t2 = generate_diverse_tasks(n_tasks=16, n_groups=4, seed=42)
        assert t1 == t2

    def test_different_seed_different_tasks(self):
        t1 = generate_diverse_tasks(n_tasks=16, n_groups=4, seed=42)
        t2 = generate_diverse_tasks(n_tasks=16, n_groups=4, seed=99)
        assert t1 != t2

    def test_group_assignment(self):
        tasks = generate_diverse_tasks(n_tasks=40, n_groups=8, seed=42)
        groups = {t["group_id"] for t in tasks}
        assert len(groups) == 8

    def test_comparison_answer_is_0_or_1(self):
        tasks = generate_diverse_tasks(n_tasks=80, n_groups=10, seed=42)
        comp = [t for t in tasks if t["subtype"] == "comparison"]
        assert len(comp) > 0
        for t in comp:
            assert t["answer"] in (0, 1)

    def test_multi_step_has_correct_answer(self):
        tasks = generate_diverse_tasks(n_tasks=80, n_groups=10, seed=42)
        ms = [t for t in tasks if t["subtype"] == "multi_step"]
        assert len(ms) > 0
        for t in ms:
            # Answer should be a reasonable integer
            assert isinstance(t["answer"], int)
            assert -1000 < t["answer"] < 10000


class TestBuildRetrievalStore:
    def test_store_has_examples_per_subtype(self):
        tasks = generate_diverse_tasks(n_tasks=80, n_groups=10, seed=42)
        store = build_retrieval_store(tasks, n_per_subtype=3)
        # 8 subtypes × 3 = 24
        assert len(store) == 24

    def test_store_entries_have_required_fields(self):
        tasks = generate_diverse_tasks(n_tasks=80, n_groups=10, seed=42)
        store = build_retrieval_store(tasks, n_per_subtype=2)
        for s in store:
            assert "prompt" in s
            assert "answer" in s
            assert "subtype" in s
