# Reliability-Aware Dyna-Q Experiment

This package tests whether scaling simulated planning updates by an empirical model-reliability estimate can reduce the damage caused by biased planning transitions.

## Algorithms

- `q_learning`: model-free Q-learning control.
- `dyna_q_fixed`: Dyna-Q with fixed epsilon = 0.2.
- `dyna_q_adaptive`: Dyna-Q with the previously tested action-value-dispersion epsilon heuristic.
- `dyna_q_reliability`: Dyna-Q with fixed epsilon = 0.2, but planning TD updates are scaled by an empirical reliability weight.

## Reliability estimator

For each visited state-action pair, the experiment compares the planning model's predicted next state with the next state observed from the real environment. The reliability estimate uses Laplace smoothing:

`reliability = (matches + 1) / (observations + 2)`

The planning update is:

`Q(s,a) <- Q(s,a) + alpha * reliability * (target - Q(s,a))`

Real-interaction Q updates are not down-weighted. In this controlled experiment, corruption is injected into planning predictions only. The agent does not receive the corruption mask as a feature; the reliability estimate is calculated from prediction/observation agreement. This is a simple heuristic, not a calibrated uncertainty estimator.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Smoke test:

```bash
python run_reliability_aware.py --seeds 2 --episodes 20 --planning-steps 5 --eval-episodes 3 --bias-levels 0,0.1,0.3
```

Full experiment:

```bash
python run_reliability_aware.py --seeds 10 --episodes 200 --planning-steps 5 --eval-episodes 10 --bias-levels 0,0.05,0.10,0.20,0.30
```

The full design runs 3 mazes x 5 bias levels x 4 algorithms x 10 seeds = 600 algorithm-condition runs. Runtime depends on the machine.

## Outputs

A timestamped folder is created under `results/` (or the folder passed with `--output`) containing:

- `config.json`: exact settings, maze layouts, algorithms, and seeds.
- `per_episode_results.csv`: episode metrics including mean reliability weight applied to planning updates.
- `greedy_evaluation_results.csv`: final greedy evaluation episodes.
- `seed_level_summary.csv`, `summary_by_maze.csv`, `pooled_summary.csv`.
- Three PNG performance plots.
- Per-run Q-tables, learned transition-model snapshots, and metadata under `artifacts/`.
- `run_complete.json` with the run status and output inventory.

## Interpretation guardrails

- The main hypothesis is that reliability-weighted planning will be more robust to model bias than standard Dyna-Q.
- This is a hypothesis, not a guaranteed outcome.
- Use the same fixed mazes, bias masks, and seed values across algorithms for paired comparisons.
- These three handcrafted mazes are not sufficient to establish broad generality.
- Do not claim a neuroscience result from these tabular RL experiments; they test a computational analogy only.
- Keep tuning separate from final evaluation before making manuscript claims.
