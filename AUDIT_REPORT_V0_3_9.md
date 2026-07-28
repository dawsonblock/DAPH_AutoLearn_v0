# AUDIT REPORT V0.3.9 — Causal Learning Loop Repair

## Architecture Before (v0.3.8 / v0.3.8.1)

The counterfactual AutoLearn loop (`CounterfactualLearningLoop` in
`trust_region.py`) had a broken causal chain:

1. **Training**: Executed both backends counterfactually, derived Delta-U
   targets, captured activations, formed a contrastive candidate direction.
   This part was correct.

2. **Evaluation (BROKEN)**: `_evaluate_held_out(candidate_vector)` accepted
   the candidate vector but **did not use it** to determine the candidate
   route. Instead:
   - The candidate route was derived from the oracle Delta-U target
     (`c_route = c_target.target`).
   - The incumbent route was **hard-coded** to `"symbolic"`
     (`i_route = "symbolic"`).

3. **Promotion**: The promotion gate operated on these invalid
   candidate-vs-incumbent comparisons, making promotion decisions
   meaningless.

4. **Lineage**: Hard-coded `layer=24`, `token_location="anchor"`,
   `alpha=1.0` regardless of actual runtime configuration.

5. **Dataset hashes**: Training and development datasets used the same hash.

6. **Capture**: Positional arrays with no task-ID alignment; shape mismatches
   could be silently repaired by truncating.

7. **Bootstrap**: Zero-vector initialization used norm-preservation, keeping
   the vector at zero forever.

8. **CLI**: Used only the legacy `run_autolearn_loop`.

9. **Versioning**: Mixed 0.3.6, 0.3.8, 0.3.8.1, 0.3.9 across surfaces.

## Architecture After (v0.3.9)

### Corrected Causal Chain

The loop now implements a genuine closed-loop learning system:

```
experience
    -> counterfactual utility (Delta-U)
    -> learning target
    -> candidate controller/vector update
    -> changed routing behavior (via RoutePolicyFn)
    -> held-out candidate-vs-incumbent evaluation
    -> promotion or rollback
    -> new incumbent
```

### Key Components

1. **RoutePolicyFn contract** (`route_policy.py`):
   - `SteeringPolicyConfig`: frozen runtime config (layer, alpha,
     token_location, model/tokenizer revision, vector version).
   - `PolicyRouteDecision`: routing decision with full provenance.
   - `make_feature_route_policy`: model-optional toy router for tests.

2. **Real candidate/incumbent evaluation** (`trust_region.py`):
   - `_evaluate_held_out(candidate_vector, incumbent_vector)` calls
     `route_fn` with each vector separately.
   - Candidate utility = utility of the backend the CANDIDATE policy selected.
   - Incumbent utility = utility of the backend the INCUMBENT policy selected.
   - Oracle Delta-U is used ONLY for the training target.

3. **Trust-region bootstrap** (`trust_region.py`):
   - `bootstrap_from_candidate`: initializes from candidate direction when
     incumbent norm < epsilon.
   - Configurable bounds: `min_vector_norm`, `max_vector_norm`,
     `max_update_norm`.
   - Fail-closed on NaN/Inf via `_validate_vector`.

4. **Task-ID-aligned capture** (`experience.py`):
   - `CapturedActivation`: per-task capture result with task_id, activation,
     success, failure_reason, activation_hash.
   - `CaptureResult`: structured capture result replacing the legacy tuple.
   - `_align_by_task_id`: joins activations and weights by task_id.

5. **Model-backed adapters** (`adapters.py`):
   - `make_model_route_policy`: wraps `route_tasks` into `RoutePolicyFn`.
   - `make_model_capture_fn`: wraps `_capture_activations` into
     `CaptureResult`.
   - `make_model_execute_fn`: wraps symbolic executor and LLM runner.

6. **CLI integration** (`cli/commands/autolearn.py`):
   - `--engine counterfactual` (default): corrected loop.
   - `--engine legacy` (deprecated): v0.3.8.1 loop with deprecation warning.

7. **Promotion gate** (`promotion.py`):
   - `min_sample_count`, `capability_regression_thresholds`,
     `protected_capabilities` added.
   - `HeldOutTaskResult` carries `candidate_route`, `incumbent_route`,
     `capability_id`, `task_family`, `utility_delta`.

8. **Versioning**: All surfaces agree on `0.3.9`.

## Remaining Limitations

1. **Real-model qualification**: The synthetic tests prove the causal chain
   is correct, but real-model qualification (GATE 13) requires running with
   an actual HuggingFace model (Qwen2.5-1.5B-Instruct). The script and
   adapters are provided but not executed in the unit-test suite.

2. **Random-direction control** (Section 20): The `random_direction_at_scale`
   function exists and is tested, but the full N>=500 random-control
   comparison pipeline is not integrated into the CLI. This is a
   qualification-time activity, not a unit-test gate.

3. **Memory integration** (Section 21): Episodic/procedural memory systems
   are preserved but not deeply integrated with the corrected loop. A stable
   replay record format is defined by the experience ledger.

4. **SAE terminology** (Section 22): `TopKFeatureSelector` is preserved as a
   contrastive feature selector. No real SAE was implemented.

5. **Legacy loop parity** (Section 12): The legacy `run_autolearn_loop` is
   preserved and marked deprecated. Full feature parity verification
   (running both engines on the same data and comparing) is a
   qualification-time activity.

## Known Scientific Uncertainties

1. **Steering vector effectiveness on real models**: The synthetic test
   proves the causal chain is mechanically correct. Whether a learned
   steering vector actually improves routing on a real transformer model
   is an empirical question that requires the real-model qualification run.

2. **Generalization**: The synthetic environment has a clean linear signal.
   Real tasks may not have such easily separable representations. The
   feature-based route policy is a toy; the real steered router uses
   model-mediated logit scoring.

3. **Statistical power**: The promotion gate uses exact paired binomial
   tests and Clopper-Pearson intervals. For small held-out sets, statistical
   power is limited. The `min_sample_count` gate helps but does not
   guarantee sufficient power for all effect sizes.

4. **Counterfactual execution cost**: The loop executes both backends on
   every training and held-out task. For large models this is expensive.
   The architecture supports this but production deployment may need
   caching or sampling strategies.
