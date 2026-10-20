# Planning-update audit (instrumented held-out validation script)

This version adds `planning_update_audit.csv` to each run. It records the planning algorithm, maze, bias level, seed, episode, selected state-action pair, predicted next state, reliability evidence counts, computed reliability estimate, shuffled estimate when applicable, applied weight, TD target, Q value before/after, and Q-value change. It also records planning attempts skipped because the learned model had no prediction. `source_integrity.json` records the SHA-256 hash of the exact script used for the run, and per-run metadata records the same hash.

**This is an audit run, not a rerun of the 1,800-run experiment.** It writes into a new output folder and does not overwrite prior results.

## Install

```bash
pip install -r requirements.txt
```

## Small audit smoke test

```bash
python run_heldout_validation.py --seeds 1 --episodes 10 --planning-steps 5 --eval-episodes 2 --bias-levels 0,0.3 --output audit_smoke
```

The smoke test still loops over all six mazes and six algorithms, but only uses one seed, ten episodes, two evaluation episodes, and two bias levels (72 runs). Inspect the new `planning_update_audit.csv` in the created run folder.

## What to verify

- `dyna_q_no_weight` has `applied_weight` = 1.0 on each update call.
- `dyna_q_reliability` applies its computed reliability estimate.
- `dyna_q_shuffled_reliability` applies the recorded shuffled estimate when a pool exists; if no pool exists, it falls back to 0.0 for the first evidence-free case.
- `q_delta` matches `alpha * applied_weight * (target - q_before)` within floating-point tolerance.
- Rows marked `skipped_no_model_prediction` have no target or applied weight and have zero Q change.
- Reliability estimate is zero when there are no prior scored predictions for the pair; the update call then has zero Q change. The current implementation does not skip that update call explicitly; it calls the update with a zero weight.

Do not append these smoke-test results to the paper's main experimental results. If the audit passes, record it as an implementation/reproducibility check.
