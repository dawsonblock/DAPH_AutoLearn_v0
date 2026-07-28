# Quarantined v0.3.7 experiment

These files are retained for auditability only. They are not a v0.3.8
benchmark, regression target, or source of licensed scientific claims.

Known protocol violations:

- the real vector was evaluated on all 50 test tasks while random directions
  were evaluated on only the first 20;
- the unequal task subsets had materially different positive-class balance;
- only five random directions were used;
- the reported z-score treated that five-sample standard deviation as a stable
  normal null;
- alpha and probe settings were selected on the test set;
- exact prompts and generator/template families crossed train, validation,
  and test boundaries;
- the run lacks a complete frozen configuration and v0.3.8 provenance bundle.

Do not cite the stored F1 difference, z-score, p-value, OOD interpretation, or
directional-causality wording as evidence. A replacement experiment must use
family-disjoint splits, identical tasks for every direction, full-sequence
route scoring, a preregistered configuration, an empirical null with at least
500 controls, paired inference, and a final split touched once.
