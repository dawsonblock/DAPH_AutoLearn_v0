"""Tests for the diverse task generator."""

from daph_learning.executive.task_generator import (
    generate_diverse_tasks,
    build_retrieval_store,
    ALL_SUBTYPES,
    RETRIEVAL_POSITIVE_SUBTYPES,
    RETRIEVAL_TRAP_SUBTYPES,
    RETRIEVAL_NEUTRAL_SUBTYPES,
)


class TestGenerateDiverseTasks:
    def test_generates_correct_count(self):
        tasks = generate_diverse_tasks(n_tasks=80, n_groups=10, seed=42)
        assert len(tasks) == 80

    def test_has_8_subtypes(self):
        tasks = generate_diverse_tasks(n_tasks=80, n_groups=10, seed=42)
        subtypes = {t["subtype"] for t in tasks}
        assert len(subtypes) == 8
        assert "simple_add" in subtypes
        assert "simple_compare" in subtypes
        assert "medium_mul" in subtypes
        assert "pattern_extend" in subtypes
        assert "large_arithmetic" in subtypes
        assert "multi_step_word" in subtypes
        assert "trap_near" in subtypes
        assert "novel_pattern" in subtypes

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

    def test_simple_compare_answer_is_0_or_1(self):
        tasks = generate_diverse_tasks(n_tasks=80, n_groups=10, seed=42)
        comp = [t for t in tasks if t["subtype"] == "simple_compare"]
        assert len(comp) > 0
        for t in comp:
            assert t["answer"] in (0, 1)

    def test_trap_near_answer_is_sum_not_product(self):
        tasks = generate_diverse_tasks(n_tasks=80, n_groups=10, seed=42)
        traps = [t for t in tasks if t["subtype"] == "trap_near"]
        assert len(traps) > 0
        for t in traps:
            # The answer should be a sum, not a product
            # Extract the two numbers from the prompt
            import re
            nums = re.findall(r"\d+", t["prompt"])
            assert len(nums) >= 2
            a, b = int(nums[0]), int(nums[1])
            assert t["answer"] == a + b
            assert t["answer"] != a * b

    def test_novel_pattern_digit_sum(self):
        tasks = generate_diverse_tasks(n_tasks=80, n_groups=10, seed=42)
        novel = [t for t in tasks if t["subtype"] == "novel_pattern"]
        assert len(novel) > 0
        for t in novel:
            assert isinstance(t["answer"], int)

    def test_pattern_extend_has_sequence(self):
        tasks = generate_diverse_tasks(n_tasks=80, n_groups=10, seed=42)
        patterns = [t for t in tasks if t["subtype"] == "pattern_extend"]
        assert len(patterns) > 0
        for t in patterns:
            assert "sequence" in t["prompt"].lower()

    def test_subtype_groups_are_disjoint(self):
        assert RETRIEVAL_POSITIVE_SUBTYPES.isdisjoint(RETRIEVAL_TRAP_SUBTYPES)
        assert RETRIEVAL_POSITIVE_SUBTYPES.isdisjoint(RETRIEVAL_NEUTRAL_SUBTYPES)
        assert RETRIEVAL_TRAP_SUBTYPES.isdisjoint(RETRIEVAL_NEUTRAL_SUBTYPES)

    def test_all_subtypes_covered(self):
        union = RETRIEVAL_POSITIVE_SUBTYPES | RETRIEVAL_TRAP_SUBTYPES | RETRIEVAL_NEUTRAL_SUBTYPES
        assert union == set(ALL_SUBTYPES)


class TestBuildRetrievalStore:
    def test_store_only_has_positive_subtypes(self):
        tasks = generate_diverse_tasks(n_tasks=80, n_groups=10, seed=42)
        store = build_retrieval_store(tasks, n_per_subtype=3)
        store_subtypes = {s["subtype"] for s in store}
        assert store_subtypes == RETRIEVAL_POSITIVE_SUBTYPES

    def test_store_has_examples_per_subtype(self):
        tasks = generate_diverse_tasks(n_tasks=80, n_groups=10, seed=42)
        store = build_retrieval_store(tasks, n_per_subtype=3)
        # 2 positive subtypes × 3 = 6
        assert len(store) == 6

    def test_store_entries_have_required_fields(self):
        tasks = generate_diverse_tasks(n_tasks=80, n_groups=10, seed=42)
        store = build_retrieval_store(tasks, n_per_subtype=2)
        for s in store:
            assert "prompt" in s
            assert "answer" in s
            assert "subtype" in s

    def test_store_does_not_contain_trap_examples(self):
        """The store should NOT have trap_near examples — the whole
        point is that retrieval misleads trap_near tasks because the
        store has multiplication examples, not trap examples."""
        tasks = generate_diverse_tasks(n_tasks=80, n_groups=10, seed=42)
        store = build_retrieval_store(tasks, n_per_subtype=5)
        store_subtypes = {s["subtype"] for s in store}
        assert "trap_near" not in store_subtypes
