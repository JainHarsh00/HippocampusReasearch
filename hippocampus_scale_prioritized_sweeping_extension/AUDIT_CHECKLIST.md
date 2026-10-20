# Pre-run and post-run audit checklist

## Before the full run
- [ ] Confirm the four static maps parse and are solvable (script checks this at import).
- [ ] Run the 56-run smoke test and confirm `run_complete.json` reports completion.
- [ ] Confirm the smoke test produces all seven algorithm labels and `paired_bootstrap_vs_reliability.csv`.
- [ ] Keep the previous result archives read-only and use a new output directory.
- [ ] Record the Python version and SHA-256 of `run_scale_ps_validation.py` with the final full-run archive.

## After the full run
- [ ] Confirm `completed_runs == planned_runs == 840` in `run_complete.json`.
- [ ] Confirm all 4 × 3 × 7 × 10 = 840 unique maze/bias/algorithm/seed conditions occur in seed-level summaries.
- [ ] Check all evaluation success rates are between 0 and 1 and all step counts are within the 250-step cap.
- [ ] Verify the 0%, 10%, and 30% bias levels are present for each method and maze.
- [ ] Verify the prioritized-sweeping method has nonzero planning updates on at least some episodes and that counts never exceed 5 per real step on average at the episode level (because planning budget is capped at five per real step).
- [ ] Inspect per-maze results; do not rely only on the pooled means.
- [ ] Check whether results change with grid size/path length and whether prioritized sweeping is more or less robust than uniform random planning.
- [ ] Report null or unfavorable results as found; do not tune methods after inspecting the held-out outcomes without labeling the changes as a new experiment.

## Caveats
- This package tests larger deterministic grids and a prioritized-sweeping baseline only.
- Stochastic transitions are not included.
- Prioritized sweeping may use fewer than five simulated updates in steps where the priority queue is empty or no pair exceeds the threshold.
- Wall-clock/runtime comparisons are not recorded as a primary metric.
