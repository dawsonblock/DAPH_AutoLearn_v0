"""V039-010 — retrievable/revisable memory + RISER-style vector library +
SAE-selected features.

Acceptance tests for Item 6:
* episodic memory is retrievable (by key, metadata, cosine similarity)
* episodic memory is revisable (revisions retained, parent chain intact)
* memory records are hashed and integrity-verifiable
* the vector library retrieves the best vector for a behavior by metric
* the SAE-style feature selector picks top-k contrastive features and
  projects onto them
"""

from __future__ import annotations

import numpy as np
import pytest

from daph_learning.memory import (
    EpisodicMemory,
    FeatureSelector,
    TopKFeatureSelector,
    VectorLibrary,
    build_record,
    make_entry,
)


# --- episodic memory: retrievable ---

def test_store_and_retrieve_by_key():
    mem = EpisodicMemory()
    rec = mem.store("task_42", {"outcome": "correct"}, embedding=[1.0, 0.0, 0.0])
    assert mem.retrieve("task_42") is rec
    assert mem.retrieve("missing") is None
    assert "task_42" in mem
    assert len(mem) == 1


def test_retrieve_by_metadata():
    mem = EpisodicMemory()
    mem.store("a", {"x": 1}, metadata={"category": "arithmetic", "split": "train"})
    mem.store("b", {"x": 2}, metadata={"category": "knowledge", "split": "train"})
    mem.store("c", {"x": 3}, metadata={"category": "arithmetic", "split": "dev"})
    hits = mem.retrieve_by_metadata(category="arithmetic")
    assert {h.key for h in hits} == {"a", "c"}
    hits = mem.retrieve_by_metadata(category="arithmetic", split="train")
    assert {h.key for h in hits} == {"a"}


def test_retrieve_similar_cosine():
    mem = EpisodicMemory()
    mem.store("a", {}, embedding=[1.0, 0.0, 0.0])
    mem.store("b", {}, embedding=[0.0, 1.0, 0.0])
    mem.store("c", {}, embedding=[0.9, 0.1, 0.0])
    # query close to a and c
    results = mem.retrieve_similar([1.0, 0.0, 0.0], limit=2)
    assert results[0][1].key == "a"
    assert results[1][1].key == "c"
    assert results[0][0] >= results[1][0]  # descending


def test_retrieve_similar_skips_records_without_embedding():
    mem = EpisodicMemory()
    mem.store("no_emb", {})
    mem.store("emb", {}, embedding=[1.0, 0.0])
    results = mem.retrieve_similar([1.0, 0.0], limit=5)
    assert [r.key for _, r in results] == ["emb"]


# --- episodic memory: revisable ---

def test_revise_retains_history_and_increments_revision():
    mem = EpisodicMemory()
    r1 = mem.store("k", {"v": 1})
    r2 = mem.revise("k", {"v": 2})
    assert r1.revision == 1
    assert r2.revision == 2
    assert r2.parent_hash == r1.record_hash
    # retrieve returns the latest revision
    assert mem.retrieve("k") is r2
    # history retains both
    history = mem.history("k")
    assert [h.revision for h in history] == [1, 2]


def test_revise_unknown_key_raises():
    mem = EpisodicMemory()
    with pytest.raises(KeyError):
        mem.revise("missing", {})


def test_revise_carries_forward_metadata():
    mem = EpisodicMemory()
    mem.store("k", {}, metadata={"category": "arithmetic", "note": "orig"})
    r2 = mem.revise("k", {}, metadata={"note": "updated"})
    assert r2.metadata["category"] == "arithmetic"  # carried forward
    assert r2.metadata["note"] == "updated"          # overridden


# --- episodic memory: hashed + integrity ---

def test_record_hash_changes_with_content():
    r1 = build_record(key="k", content={"v": 1})
    r2 = build_record(key="k", content={"v": 2})
    assert r1.record_hash != r2.record_hash


def test_verify_integrity_passes_on_clean_store():
    mem = EpisodicMemory()
    mem.store("k1", {"v": 1})
    mem.store("k2", {"v": 2}, metadata={"cat": "x"})
    mem.revise("k1", {"v": 1.1})
    mem.verify_integrity()  # must not raise


def test_verify_integrity_detects_parent_chain_break():
    mem = EpisodicMemory()
    mem.store("k", {"v": 1})
    # append a self-consistent revision whose parent_hash is wrong, so the
    # hash check passes but the parent-chain check fires.
    recs = mem._records["k"]  # type: ignore[attr-defined]
    bad = build_record(key="k", content={"v": 2}, revision=2,
                       parent_hash="tampered")
    recs.append(bad)
    with pytest.raises(ValueError, match="parent_hash chain broken"):
        mem.verify_integrity()


# --- RISER-style vector library ---

def test_vector_library_retrieves_best_for_behavior():
    lib = VectorLibrary()
    lib.add(make_entry(vector_id="v1", behavior="invoke_symbolic_tool",
                       family="tool_policy", layer=24, alpha=1.0,
                       values=[0.1, 0.2], metrics={"held_out_utility": 0.5}))
    lib.add(make_entry(vector_id="v2", behavior="invoke_symbolic_tool",
                       family="tool_policy", layer=24, alpha=1.0,
                       values=[0.3, 0.4], metrics={"held_out_utility": 0.9}))
    lib.add(make_entry(vector_id="v3", behavior="reasoning_policy",
                       family="reasoning_policy", layer=16, alpha=1.0,
                       values=[0.5, 0.6], metrics={"held_out_utility": 0.99}))
    best = lib.best_for_behavior("invoke_symbolic_tool")
    assert best is not None
    assert best.vector_id == "v2"  # higher utility


def test_vector_library_filters_by_layer_and_family():
    lib = VectorLibrary()
    lib.add(make_entry(vector_id="v1", behavior="b", family="tool_policy",
                       layer=24, alpha=1.0, values=[0.1], metrics={"held_out_utility": 0.5}))
    lib.add(make_entry(vector_id="v2", behavior="b", family="tool_policy",
                       layer=16, alpha=1.0, values=[0.2], metrics={"held_out_utility": 0.9}))
    hits = lib.retrieve(behavior="b", family="tool_policy", layer=16)
    assert [h.vector_id for h in hits] == ["v2"]


def test_vector_entry_hashed():
    e1 = make_entry(vector_id="v1", behavior="b", family="f", layer=24,
                    alpha=1.0, values=[0.1, 0.2])
    e2 = make_entry(vector_id="v1", behavior="b", family="f", layer=24,
                    alpha=1.0, values=[0.1, 0.3])
    assert e1.vector_hash != e2.vector_hash


# --- SAE-selected features ---

def test_topk_selector_implements_protocol():
    sel = TopKFeatureSelector(k=2)
    assert isinstance(sel, FeatureSelector)


def test_topk_selector_picks_contrastive_features():
    sel = TopKFeatureSelector(k=2)
    # feature 0 separates the classes strongly; feature 2 also; feature 1 is noise
    acts = np.array([
        [10.0, 0.1, 0.0],
        [10.0, 0.2, 0.1],
        [0.0, 0.0, 10.0],
        [0.1, 0.1, 10.0],
    ], dtype=np.float32)
    labels = [1, 1, 0, 0]
    sel.fit(acts, labels)
    selected = sel.select(k=2)
    # features 0 and 2 have the largest mean-diff
    assert set(selected.tolist()) == {0, 2}


def test_topk_selector_project_onto_subspace():
    sel = TopKFeatureSelector(k=2)
    acts = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    sel.fit(np.array([[1.0, 0.0, 9.0, 0.0],
                      [0.0, 1.0, 0.0, 9.0]], dtype=np.float32), [1, 0])
    selected = sel.select(k=2)
    projected = sel.project(acts)
    assert projected.shape == (1, 2)
    # projected columns correspond to the selected indices
    np.testing.assert_allclose(projected[0], acts[0, selected])


def test_topk_selector_requires_fit():
    sel = TopKFeatureSelector(k=2)
    with pytest.raises(ValueError, match="not fitted"):
        sel.select()


def test_topk_selector_rejects_non_binary_labels():
    sel = TopKFeatureSelector(k=2)
    with pytest.raises(ValueError, match="binary"):
        sel.fit(np.zeros((3, 2), dtype=np.float32), [0, 1, 2])
