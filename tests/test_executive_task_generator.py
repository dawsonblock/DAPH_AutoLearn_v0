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
        tasks = generate_diverse_tasks(n_tasks=90, n_groups=10, seed=42)
        assert len(tasks) == 90

    def test_has_9_subtypes(self):
        tasks = generate_diverse_tasks(n_tasks=90, n_groups=10, seed=42)
        subtypes = {t["subtype"] for t in tasks}
        assert len(subtypes) == 9
        assert "simple_add" in subtypes
        assert "simple_compare" in subtypes
        assert "digit_manip" in subtypes
        assert "pattern_extend" in subtypes
        assert "formula_apply" in subtypes
        assert "multi_step_arith" in subtypes
        assert "multi_step_word" in subtypes
        assert "hard_mul" in subtypes
        assert "trap_near" in subtypes

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
        tasks = generate_diverse_tasks(n_tasks=90, n_groups=10, seed=42)
        comp = [t for t in tasks if t["subtype"] == "simple_compare"]
        assert len(comp) > 0
        for t in comp:
            assert t["answer"] in (0, 1)

    def test_trap_near_answer_is_sum_not_product(self):
        tasks = generate_diverse_tasks(n_tasks=90, n_groups=10, seed=42)
        traps = [t for t in tasks if t["subtype"] == "trap_near"]
        assert len(traps) > 0
        for t in traps:
            # The answer should be a sum, not a product
            import re
            nums = re.findall(r"\d+", t["prompt"])
            assert len(nums) >= 2
            a, b = int(nums[0]), int(nums[1])
            assert t["answer"] == a + b
            assert t["answer"] != a * b

    def test_digit_manip_has_digit_operation(self):
        tasks = generate_diverse_tasks(n_tasks=90, n_groups=10, seed=42)
        digit_tasks = [t for t in tasks if t["subtype"] == "digit_manip"]
        assert len(digit_tasks) > 0
        for t in digit_tasks:
            assert isinstance(t["answer"], int)

    def test_pattern_extend_has_sequence(self):
        tasks = generate_diverse_tasks(n_tasks=90, n_groups=10, seed=42)
        patterns = [t for t in tasks if t["subtype"] == "pattern_extend"]
        assert len(patterns) > 0
        for t in patterns:
            assert "sequence" in t["prompt"].lower()

    def test_formula_apply_has_formula_keyword(self):
        tasks = generate_diverse_tasks(n_tasks=90, n_groups=10, seed=42)
        formulas = [t for t in tasks if t["subtype"] == "formula_apply"]
        assert len(formulas) > 0
        for t in formulas:
            assert isinstance(t["answer"], int)

    def test_hard_mul_is_two_digit(self):
        tasks = generate_diverse_tasks(n_tasks=90, n_groups=10, seed=42)
        hards = [t for t in tasks if t["subtype"] == "hard_mul"]
        assert len(hards) > 0
        for t in hards:
            assert "×" in t["prompt"]

    def test_multi_step_arith_has_multiple_steps(self):
        tasks = generate_diverse_tasks(n_tasks=90, n_groups=10, seed=42)
        multi = [t for t in tasks if t["subtype"] == "multi_step_arith"]
        assert len(multi) > 0
        for t in multi:
            assert "First calculate" in t["prompt"]

    def test_subtype_groups_are_disjoint(self):
        assert RETRIEVAL_POSITIVE_SUBTYPES.isdisjoint(RETRIEVAL_TRAP_SUBTYPES)
        assert RETRIEVAL_POSITIVE_SUBTYPES.isdisjoint(RETRIEVAL_NEUTRAL_SUBTYPES)
        assert RETRIEVAL_TRAP_SUBTYPES.isdisjoint(RETRIEVAL_NEUTRAL_SUBTYPES)

    def test_all_subtypes_covered(self):
        union = RETRIEVAL_POSITIVE_SUBTYPES | RETRIEVAL_TRAP_SUBTYPES | RETRIEVAL_NEUTRAL_SUBTYPES
        assert union == set(ALL_SUBTYPES)


class TestBuildRetrievalStore:
    def test_store_only_has_positive_subtypes(self):
        tasks = generate_diverse_tasks(n_tasks=90, n_groups=10, seed=42)
        store = build_retrieval_store(tasks, n_per_subtype=3)
        store_subtypes = {s["subtype"] for s in store}
        assert store_subtypes == RETRIEVAL_POSITIVE_SUBTYPES

    def test_store_has_examples_per_subtype(self):
        tasks = generate_diverse_tasks(n_tasks=90, n_groups=10, seed=42)
        store = build_retrieval_store(tasks, n_per_subtype=3)
        # 2 positive subtypes × 3 = 6
        assert len(store) == 6

    def test_store_entries_have_required_fields(self):
        tasks = generate_diverse_tasks(n_tasks=90, n_groups=10, seed=42)
        store = build_retrieval_store(tasks, n_per_subtype=2)
        for s in store:
            assert "prompt" in s
            assert "answer" in s
            assert "subtype" in s

    def test_store_does_not_contain_trap_examples(self):
        """The store should NOT have trap_near examples — the whole
        point is that retrieval misleads trap_near tasks because the
        store has multiplication examples, not trap examples."""
        tasks = generate_diverse_tasks(n_tasks=90, n_groups=10, seed=42)
        store = build_retrieval_store(tasks, n_per_subtype=5)
        store_subtypes = {s["subtype"] for s in store}
        assert "trap_near" not in store_subtypes
