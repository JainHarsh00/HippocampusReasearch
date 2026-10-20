# Controlled Model-Imperfection Experiment

This experiment tests whether Dyna-style planning changes performance when its internal transition model is accurate, the environment is stochastic, the planning model is incomplete, or selected planning transitions are biased.

## Conditions

- `accurate_deterministic`: deterministic maze, unmodified planning model.
- `stochastic_transitions`: with probability `--slip-probability`, the environment executes a random action instead of the chosen action. The Dyna model stores observed outcomes for each state-action pair, so this is a deliberately simple empirical model under stochastic dynamics.
- `incomplete_model`: a fixed fraction of first-observed state-action pairs are excluded from planning updates. They remain available for real experience updates.
- `biased_model`: a fixed fraction of first-observed state-action pairs receive an incorrect predicted next state during planning updates only. Real updates remain correct.

## Algorithms

- `q_learning_fixed`: model-free baseline.
- `dyna_q_fixed`: Dyna-Q with fixed epsilon = 0.2.
- `dyna_q_adaptive`: Dyna-Q with the same Q-dispersion adaptive-epsilon heuristic used in the earlier diagnostic (lambda = 5). It is a heuristic, not calibrated uncertainty.

## Setup and run

```bash
pip install -r requirements.txt
python run_model_imperfection.py --seeds 2 --episodes 20 --planning-steps 5 --eval-episodes 3
python run_model_imperfection.py --seeds 10 --episodes 200 --planning-steps 5 --eval-episodes 10
```

Optional controlled impairment parameters:

```bash
python run_model_imperfection.py --seeds 10 --episodes 200 --slip-probability 0.15 --incomplete-fraction 0.25 --biased-fraction 0.20
```

## Saved outputs

Each invocation creates a timestamped results directory with the exact configuration, all episode records, evaluation records, per-seed summaries, aggregate summaries, plots, Q-tables, learned transition models, and metadata for every seed/condition/algorithm. Results are never intentionally written into an existing run folder.

## Interpretation cautions

- Use seed-level summaries as the independent replicates; episodes within one seed are not independent samples.
- The same maze is used across conditions. Results are a controlled diagnostic, not evidence of generalization across environments.
- The `biased_model` condition is an explicit intervention that corrupts next-state predictions for a subset of pairs; it is not claimed to represent a biologically faithful hippocampal model.
- Do not edit the manuscript with invented or unverified results. Report what the output files show, including negative results.
