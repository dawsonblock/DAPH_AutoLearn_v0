"""Tests for DAPH v0.4 executive action executors.

Tests the DirectReasoningExecutor, RetrievalVectorExecutor,
ReasoningDecomposeExecutor, and ExecutorRegistry with mock generators.
"""

from __future__ import annotations

import pytest
import numpy as np

from daph_learning.executive import (
    ActionDescriptor,
    ActionSpace,
    ActionExecution,
    CounterfactualSet,
    UtilityModel,
    ExecutiveState,
    LLMGenerationConfig,
    DirectReasoningExecutor,
    RetrievalVectorExecutor,
    ReasoningDecomposeExecutor,
    ExecutorRegistry,
    build_b1_executors,
    build_executive_experiences,
    evaluate_qualification,
    ExecutiveTaskRecord,
    ExecutiveLogisticPolicy,
)


# ──────────────────────────────────────────────────────────────────────
# Mock Generators
# ──────────────────────────────────────────────────────────────────────

def _make_mock_generator(correct_answer: int):
    """Create a mock generate_fn that always returns the correct answer."""
    def gen(prompt, config):
        return f"FINAL_ANSWER: {correct_answer}", 50.0
    return gen


def _make_wrong_generator():
    """Create a mock generate_fn that always returns a wrong answer."""
    def gen(prompt, config):
        return "FINAL_ANSWER: 99999", 50.0
    return gen


def _make_error_generator():
    """Create a mock generate_fn that always errors."""
    def gen(prompt, config):
        return "ERROR: connection failed", 10.0
    return gen


def _make_parse_error_generator():
    """Create a mock generate_fn that returns unparseable text."""
    def gen(prompt, config):
        return "I don't know the answer", 30.0
    return gen


# ──────────────────────────────────────────────────────────────────────
# DirectReasoningExecutor Tests
# ──────────────────────────────────────────────────────────────────────

class TestDirectReasoningExecutor:
    def test_correct_answer(self):
        gen = _make_mock_generator(42)
        executor = DirectReasoningExecutor(generate_fn=gen)
        task = {"task_id": "t1", "prompt": "What is 40+2?", "answer": 42}
        result = executor.execute(task)
        assert result.action_id == "action.reasoning.direct"
        assert result.executed is True
        assert result.verified_correct is True
        assert result.failure_type is None
        assert "FINAL_ANSWER: 42" in result.output

    def test_wrong_answer(self):
        gen = _make_wrong_generator()
        executor = DirectReasoningExecutor(generate_fn=gen)
        task = {"task_id": "t1", "prompt": "What is 40+2?", "answer": 42}
        result = executor.execute(task)
        assert result.verified_correct is False
        assert result.failure_type is None  # wrong answer, not execution failure

    def test_execution_error(self):
        gen = _make_error_generator()
        executor = DirectReasoningExecutor(generate_fn=gen)
        task = {"task_id": "t1", "prompt": "What is 40+2?", "answer": 42}
        result = executor.execute(task)
        assert result.verified_correct is False
        assert result.failure_type == "execution_error"

    def test_parse_error(self):
        gen = _make_parse_error_generator()
        executor = DirectReasoningExecutor(generate_fn=gen)
        task = {"task_id": "t1", "prompt": "What is 40+2?", "answer": 42}
        result = executor.execute(task)
        assert result.verified_correct is False
        assert result.failure_type == "parse_error"

    def test_no_answer_field(self):
        gen = _make_mock_generator(42)
        executor = DirectReasoningExecutor(generate_fn=gen)
        task = {"task_id": "t1", "prompt": "What is 40+2?"}
        result = executor.execute(task)
        assert result.verified_correct is None  # unverifiable

    def test_cost_estimate(self):
        gen = _make_mock_generator(42)
        executor = DirectReasoningExecutor(generate_fn=gen, cost_estimate=0.15)
        task = {"task_id": "t1", "prompt": "x", "answer": 42}
        result = executor.execute(task)
        assert result.compute_cost == 0.15


# ──────────────────────────────────────────────────────────────────────
# RetrievalVectorExecutor Tests
# ──────────────────────────────────────────────────────────────────────

class TestRetrievalVectorExecutor:
    def test_correct_with_examples(self):
        gen = _make_mock_generator(42)
        examples = [
            {"prompt": "What is 30+12?", "answer": 42},
            {"prompt": "What is 20+22?", "answer": 42},
        ]
        executor = RetrievalVectorExecutor(
            generate_fn=gen, examples=examples, n_retrieved=2)
        task = {"task_id": "t1", "prompt": "What is 40+2?", "answer": 42}
        result = executor.execute(task)
        assert result.action_id == "action.retrieval.vector"
        assert result.verified_correct is True

    def test_prompt_includes_examples(self):
        """The generated prompt should include retrieved examples."""
        captured_prompt = []
        def capturing_gen(prompt, config):
            captured_prompt.append(prompt)
            return "FINAL_ANSWER: 42", 50.0
        examples = [{"prompt": "ex1", "answer": 10}]
        executor = RetrievalVectorExecutor(
            generate_fn=capturing_gen, examples=examples, n_retrieved=1)
        task = {"task_id": "t1", "prompt": "test problem", "answer": 42}
        executor.execute(task)
        assert len(captured_prompt) == 1
        assert "similar examples" in captured_prompt[0]
        assert "ex1" in captured_prompt[0]

    def test_empty_examples(self):
        gen = _make_mock_generator(42)
        executor = RetrievalVectorExecutor(
            generate_fn=gen, examples=[], n_retrieved=3)
        task = {"task_id": "t1", "prompt": "x", "answer": 42}
        result = executor.execute(task)
        assert result.verified_correct is True

    def test_custom_retrieve_fn(self):
        gen = _make_mock_generator(42)
        retrieved = []
        def my_retrieve(query, examples, k):
            retrieved.append(query)
            return examples[:k]
        executor = RetrievalVectorExecutor(
            generate_fn=gen, examples=[{"prompt": "a", "answer": 1}],
            retrieve_fn=my_retrieve, n_retrieved=1)
        task = {"task_id": "t1", "prompt": "x", "answer": 42}
        executor.execute(task)
        assert len(retrieved) == 1


# ──────────────────────────────────────────────────────────────────────
# ReasoningDecomposeExecutor Tests
# ──────────────────────────────────────────────────────────────────────

class TestReasoningDecomposeExecutor:
    def test_correct_answer(self):
        gen = _make_mock_generator(42)
        executor = ReasoningDecomposeExecutor(generate_fn=gen)
        task = {"task_id": "t1", "prompt": "What is 40+2?", "answer": 42}
        result = executor.execute(task)
        assert result.action_id == "action.reasoning.decompose"
        assert result.verified_correct is True

    def test_decompose_prompt_contains_instruction(self):
        captured = []
        def capturing_gen(prompt, config):
            captured.append(prompt)
            return "FINAL_ANSWER: 42", 100.0
        executor = ReasoningDecomposeExecutor(generate_fn=capturing_gen)
        task = {"task_id": "t1", "prompt": "Solve x", "answer": 42}
        executor.execute(task)
        assert "SUB:" in captured[0]
        assert "decompose" in captured[0].lower() or "Break" in captured[0]

    def test_wrong_answer(self):
        gen = _make_wrong_generator()
        executor = ReasoningDecomposeExecutor(generate_fn=gen)
        task = {"task_id": "t1", "prompt": "x", "answer": 42}
        result = executor.execute(task)
        assert result.verified_correct is False

    def test_cost_estimate_higher(self):
        gen = _make_mock_generator(42)
        executor = ReasoningDecomposeExecutor(generate_fn=gen, cost_estimate=0.30)
        task = {"task_id": "t1", "prompt": "x", "answer": 42}
        result = executor.execute(task)
        assert result.compute_cost == 0.30


# ──────────────────────────────────────────────────────────────────────
# ExecutorRegistry Tests
# ──────────────────────────────────────────────────────────────────────

class TestExecutorRegistry:
    def test_register_and_get(self):
        registry = ExecutorRegistry()
        executor = DirectReasoningExecutor(generate_fn=_make_mock_generator(42))
        registry.register(executor)
        assert registry.get("action.reasoning.direct") is executor

    def test_get_missing(self):
        registry = ExecutorRegistry()
        assert registry.get("nonexistent") is None

    def test_execute_all(self):
        gen = _make_mock_generator(42)
        registry = build_b1_executors(
            LLMGenerationConfig(),
            generate_fn=gen,
            examples=[{"prompt": "ex", "answer": 1}],
        )
        space = ActionSpace(actions=(
            ActionDescriptor(action_id="action.reasoning.direct"),
            ActionDescriptor(action_id="action.retrieval.vector"),
            ActionDescriptor(action_id="action.reasoning.decompose"),
        ))
        task = {"task_id": "t1", "prompt": "What is 40+2?", "answer": 42}
        cf_set = registry.execute_all(task, space)
        assert isinstance(cf_set, CounterfactualSet)
        assert cf_set.task_id == "t1"
        assert len(cf_set.executions) == 3
        assert cf_set.all_executed is True
        for exec in cf_set.executions.values():
            assert exec.verified_correct is True

    def test_execute_all_missing_executor(self):
        registry = ExecutorRegistry()
        space = ActionSpace(actions=(
            ActionDescriptor(action_id="action.reasoning.direct"),
            ActionDescriptor(action_id="action.retrieval.vector"),
        ))
        task = {"task_id": "t1", "prompt": "x", "answer": 42}
        cf_set = registry.execute_all(task, space)
        assert cf_set.all_executed is False
        for exec in cf_set.executions.values():
            assert exec.executed is False
            assert exec.failure_type == "no_executor"


# ──────────────────────────────────────────────────────────────────────
# B1 Factory Tests
# ──────────────────────────────────────────────────────────────────────

class TestB1Factory:
    def test_build_b1_executors(self):
        registry = build_b1_executors(
            LLMGenerationConfig(model_id="test-model"),
            generate_fn=_make_mock_generator(42),
        )
        assert "action.reasoning.direct" in registry.action_ids
        assert "action.retrieval.vector" in registry.action_ids
        assert "action.reasoning.decompose" in registry.action_ids

    def test_b1_end_to_end_with_mock(self):
        """Full B1 pipeline with mock generators: exec → experience → policy → eval."""
        # Create mock generators with different correctness patterns
        # Direct: correct on cluster 0
        # Retrieval: correct on cluster 1
        # Decompose: correct on cluster 2
        def cluster_aware_gen(prompt, config):
            # Determine cluster from prompt
            if "cluster_0" in prompt:
                return "FINAL_ANSWER: 0", 50.0
            elif "cluster_1" in prompt:
                return "FINAL_ANSWER: 1", 100.0
            elif "cluster_2" in prompt:
                return "FINAL_ANSWER: 2", 200.0
            return "FINAL_ANSWER: 99", 50.0

        # Actually, we need different generators per action.
        # Let's use a simpler approach: all correct.
        gen = _make_mock_generator(42)
        registry = build_b1_executors(
            LLMGenerationConfig(model_id="test"),
            generate_fn=gen,
            examples=[{"prompt": "ex", "answer": 42}],
        )

        space = ActionSpace(actions=(
            ActionDescriptor(action_id="action.reasoning.direct", cost_estimate=0.15),
            ActionDescriptor(action_id="action.retrieval.vector", cost_estimate=0.10),
            ActionDescriptor(action_id="action.reasoning.decompose", cost_estimate=0.30),
        ))

        # Generate tasks
        tasks = []
        for i in range(30):
            tasks.append({
                "task_id": f"t{i}",
                "prompt": f"What is 40+2? (task {i})",
                "answer": 42,
                "subtype": "A" if i % 2 == 0 else "B",
                "group_id": f"g{i % 5}",
            })

        # Execute all actions on all tasks
        cf_sets = [registry.execute_all(task, space) for task in tasks]

        # Build experiences
        um = UtilityModel(
            quality_weight=1.0, lambda_time=0.01,
            lambda_compute=0.1, lambda_risk=1.0,
        )
        experiences = build_executive_experiences(cf_sets, um, space)

        # Extract features (random for this test)
        rng = np.random.RandomState(42)
        features = rng.randn(30, 5).astype(np.float32)

        # Train policy
        from daph_learning.executive import experiences_to_training_arrays
        arrays = experiences_to_training_arrays(experiences, space, features=features)
        policy = ExecutiveLogisticPolicy()
        policy.fit(
            arrays["features"], arrays["utilities"],
            arrays["weights"], space, n_iter=200)

        # Predict
        probs = policy.predict_proba(features)

        # Build task records
        records = []
        for i, cf in enumerate(cf_sets):
            p = probs[i]
            sel_idx = int(np.argmax(p))
            oracle_idx = int(np.argmax(arrays["utilities"][i]))
            records.append(ExecutiveTaskRecord(
                task_id=cf.task_id,
                group_id=cf.state.group_id or "g0",
                subtype=cf.state.subtype or "A",
                split="test",
                utilities={space.action_ids[j]: float(arrays["utilities"][i, j]) for j in range(3)},
                probabilities={space.action_ids[j]: float(p[j]) for j in range(3)},
                selected_action=space.action_ids[sel_idx],
                oracle_action=space.action_ids[oracle_idx],
                p1_realized_utility=float(arrays["utilities"][i, sel_idx]),
                p0_realized_utility=float(arrays["utilities"][i, 0]),
                oracle_utility=float(arrays["utilities"][i, oracle_idx]),
                regret=float(arrays["utilities"][i, oracle_idx] - arrays["utilities"][i, sel_idx]),
            ))

        # Evaluate
        result = evaluate_qualification(
            records, space,
            experiment_id="b1_mock_test",
            bootstrap_iterations=500,
        )
        assert result.n_tasks == 30
        assert len(result.always_action_utilities) == 3
