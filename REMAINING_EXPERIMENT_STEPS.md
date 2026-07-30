# Remaining Experiment Steps — DAPH AutoLearn v0.3.10.4-alpha

This document states exactly what hardware, model download, runtime, and
commands are needed to take the repository from its current state
(`Gate A status: NOT YET_REQUALIFIED`, Priority 0 complete) through a full
frozen real-model Gate A requalification for `daph_gate_a_real_002`.

It is the honest scope boundary: Priority 0 (Sections 1–7) is implemented
and tested; Priority 1+ (Sections 8–27) is not yet implemented and is
required before any Gate A claim can be made.

---

## 1. Implementation work still required (Priority 1+, Sections 8–19)

Before any real-model final run, the following must be implemented, frozen,
and unit-tested. None of these are present in the current source tree.

1. **Frozen utility contract (Section 8).** A frozen `UtilityConfig`
   dataclass with `compute_utility()` and a `utility_config_hash`; two
   explicitly separated protocols (`gate_a_accuracy_primary` as the
   authoritative primary endpoint, `gate_a_cost_sensitive_secondary` as
   the secondary). The utility config must be frozen before
   development-model selection and final evaluation.
2. **Benchmark redesign for genuine crossover (Section 9).** At least 60
   (preferred 80) independent generation groups; within-subtype crossover
   for ≥3 subtypes with `min_backend_win_fraction ≥ 0.20`; crossover must
   emerge from real task properties, not label balancing.
3. **Group-before-instance splitting (Section 9.4).** Define groups →
   assign groups to splits (train 50% / dev 20% / cal 15% / final 15%) →
   generate instances within splits → deduplicate globally → validate no
   normalized prompt overlap.
4. **Leakage and benchmark audit (Section 10).** `src/daph_learning/benchmark/audit.py`
   with `DatasetAudit`; abort before model execution on group cross-split
   leaks, exact/normalized duplicates, template leaks, insufficient
   crossover, or low decisive fraction.
5. **Latency measurement (Section 11).** `TimingProtocol` and
   `measure_backend_latency()` with CUDA sync, warm-up, median
   aggregation; separate answer generation from benchmark timing.
6. **Representation-selection protocol (Section 12).** Development-only
   study over candidate layers (0.25/0.50/0.75/last) and pooling methods
   (last prompt token, mean prompt tokens, mean content tokens minimum);
   select on a frozen development metric; record every candidate; freeze
   layer/pooling/PCA/policy class/regularization/thresholds.
7. **Surface and structured-feature baselines (Section 13).** `always_llm`,
   `always_symbolic`, `hand_router`, `subtype_only_logistic`,
   `surface_tfidf_logistic`, `structured_feature_logistic`,
   `hidden_state_centroid`, `hidden_state_logistic`,
   `sham_hidden_state_logistic`, `oracle`.
8. **Uncertainty-aware training targets (Section 14).** `TargetMode` enum
   (HARD_GAP / SOFT_TEMPERATURE / SIGNAL_TO_NOISE); select mode on
   development data and freeze.
9. **Multi-seed sham control (Section 15).** Shuffle within
   `subtype × split × decisive/nondecisive`; ≥20 sham seeds; report sham
   distribution and group-aware P1-minus-sham interval.
10. **Final-access enforcement (Section 16).** Immutable `ExperimentStage`
    state machine; final-access ledger; refuse second access / source /
    dataset / policy / utility / calibration changes.
11. **Statistical qualification (Section 17).** Group-aware bootstrap
    (resample groups, not rows), `group_weighted` primary estimand,
    permutation test, leave-one-group-out, cluster-robust SE,
    subtype-stratified bootstrap, worst-subtype delta,
    positive-group fraction.
12. **Frozen gate criteria (Section 18).** `configs/gate_a_real_002.yaml`
    with primary endpoint, gates, dataset minimums, and evidence
    requirements. Values frozen before final access.
13. **Report generator (Section 19).** One generator that reads only
    validated artifacts and emits the full required output set
    (`experiment_results.json`, `gate_decision.json`,
    `bootstrap_samples.*`, `predictions.parquet`, `task_outcomes.parquet`,
    `dataset_audit.json`, `representation_selection.json`,
    `calibration.json`, `policy_manifest.json`, `freeze_manifest.json`,
    `final_access_ledger.json`, `environment.json`,
    `source_manifest.json`, `test_report.json`, `GATE_A_RESULTS.md`).
14. **Symbolic/LLM output contract alignment.** The smoke found that
    neither the symbolic backend's bare-integer output nor the LLM's prose
    uses the canonical `FINAL_ANSWER: <integer>` field, so the canonical
    verifier marks them UNVERIFIABLE. Before the real run, both backends
    must emit the canonical field (or the symbolic backend must wrap its
    output as `FINAL_ANSWER: <int>` and the LLM prompt must require it).

---

## 2. Hardware and model requirements for the full Gate A run

| Requirement | Value |
|-------------|-------|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` (float16) — the model used by the archived failed run; must be re-downloaded if not cached. A larger model may be chosen but must be frozen before final access. |
| Accelerator | A CUDA GPU with ≥16 GB VRAM is strongly recommended (the archived run used an RTX 5090, 32 GB). MPS/CPU works for the smoke but is too slow for the full final split and does not support `torch.cuda.synchronize()`. |
| Disk | ~3 GB for the 1.5B model weights + artifact bundles (parquet/npz). |
| Runtime estimate | Not estimated here (per the project's policy, no concrete timeline is given). The full run executes collect → develop → calibrate → freeze → final over ≥60 groups and a 10,000–20,000-iteration group bootstrap. |
| Network | Hugging Face Hub access to download the chosen model revision (unless already cached). The smoke ran fully offline against a cached 0.5B model. |

---

## 3. Required execution commands (once Priority 1+ is implemented)

These match the Section 22 contract. They will not succeed until the
Priority 1+ implementation above is complete.

```bash
# 0. (one-time) freeze the canonical source hash and confirm
python -m daph_learning.provenance source-hash

# 1. Collect counterfactual experience
python scripts/run_gate_a_experiment.py \
  --config configs/gate_a_real_002.yaml \
  --stage collect

# 2. Development (representation selection + target mode)
python scripts/run_gate_a_experiment.py \
  --config configs/gate_a_real_002.yaml \
  --stage develop

# 3. Calibrate thresholds (calibration data only)
python scripts/run_gate_a_experiment.py \
  --config configs/gate_a_real_002.yaml \
  --stage calibrate

# 4. Freeze the protocol and policy (one-shot, ledgered)
python scripts/freeze_gate_a.py \
  --config configs/gate_a_real_002.yaml

# 5. Final evaluation (one-shot; refuses if anything changed)
python scripts/run_gate_a_experiment.py \
  --config configs/gate_a_real_002.yaml \
  --stage final

# 6. Validate and report (path chosen by the gate decision)
python scripts/validate_gate_a_bundle.py \
  artifacts/gate_a_failed/daph_gate_a_real_002
#   or, only if every preregistered gate passed:
python scripts/validate_gate_a_bundle.py \
  artifacts/gate_a_qualified/daph_gate_a_real_002
```

The execution script must decide the final directory (`gate_a_failed/`
vs `gate_a_qualified/`) only after computing the gate decision. The
`current/pointer.json` is then regenerated to point at the resulting
bundle with `status: PASS` or `status: FAILED`.

---

## 4. Gate A pass criterion (frozen, must not change after final access)

Per Section 18, Gate A passes **only if every** preregistered criterion
passes, including:

- `require_lcb_vs_p0_above: 0.0` (group-aware 95% LCB for P1−P0 > 0);
- `require_lcb_vs_sham_above: 0.0`;
- `minimum_oracle_gap_capture: 0.50`;
- `minimum_positive_group_fraction: 0.60`;
- `maximum_final_access_count: 1`;
- `require_real_model: true`, `allow_synthetic: false`;
- `require_source_hash_match: true`, `require_dataset_hashes: true`,
  `require_final_access_ledger: true`.

**Do not claim Gate A passed unless the full frozen real-model
experiment actually ran and every preregistered gate passed.**

---

## 5. Current state summary

- Priority 0 (Sections 1–7): **implemented and tested** (1048 passed,
  4 skipped).
- Real-model smoke: **executed** (Qwen2.5-0.5B-Instruct, MPS, offline);
  bundle validated; NOT Gate A evidence.
- Full Gate A experiment: **not executed**.
- Current pointer: `NOT_YET_REQUALIFIED`.
- Old failed run: **archived** under `artifacts/legacy/daph_gate_a_real_001_failed/`.
- No synthetic artifact is presented as Gate A qualification evidence.
