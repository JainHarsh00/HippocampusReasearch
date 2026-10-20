# Audited Controlled Model-Imperfection Experiment

This is the corrected implementation following an audit of the initial run. Read `AUDIT_REPORT.md` before interpreting results. **This version changes the experiment implementation, so run it from scratch and do not combine its results with the earlier run.**

## Conditions

- `accurate_deterministic`: deterministic maze, no planning-model impairment.
- `stochastic_transitions`: with probability `--slip-probability`, the environment executes a uniformly random action instead of the selected action. Dyna stores observed outcomes for each state-action pair.
- `incomplete_model`: a preselected subset of valid state-action pairs is excluded from planning updates. Real interaction updates remain available.
- `biased_model`: a preselected subset of state-action pairs receives an incorrect nonterminal predicted next state during planning updates only. Real interaction updates remain correct.

## Algorithms

- `q_learning_fixed`: model-free baseline.
- `dyna_q_fixed`: Dyna-Q with fixed epsilon = 0.2.
- `dyna_q_adaptive`: Dyna-Q with the Q-dispersion adaptive-epsilon heuristic (lambda = 5). It is a heuristic, not calibrated uncertainty.

## Setup and run

```bash
pip install -r requirements.txt
python run_model_imperfection.py --seeds 2 --episodes 20 --planning-steps 5 --eval-episodes 3
python run_model_imperfection.py --seeds 10 --episodes 200 --planning-steps 5 --eval-episodes 10
```

Optional settings:

```bash
python run_model_imperfection.py --seeds 10 --episodes 200 --slip-probability 0.15 --incomplete-fraction 0.25 --biased-fraction 0.20
```

## Reproducibility and outputs

Separate reproducible RNG streams are used for agent decisions/planning, training environment transitions, and evaluation environment transitions. Impairment masks are precomputed per seed, independently of algorithm visitation. Each invocation creates a new timestamped results directory with configuration, episode and evaluation CSVs, seed-level and aggregate summaries, plots, Q-tables, learned-model artifacts, and metadata. Per-episode records report attempted, applied, and skipped planning updates.

## Interpretation cautions

- Seed-level summaries are the replication unit; episodes within one seed are not independent replicates.
- All conditions use one maze. Results are diagnostic, not evidence of generalization.
- The biased-model condition is a synthetic intervention, not a biologically faithful hippocampal model.
- The incomplete-model condition reduces effective planning updates for masked pairs; it is not an equal-compute comparison.
- Report negative results as well as positive results. Do not edit manuscript claims until the audited results are analyzed.
