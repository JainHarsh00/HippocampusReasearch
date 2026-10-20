# Larger-grid validation and prioritized-sweeping baseline

This is a **separate experiment package**. It does not overwrite or modify the previously completed 1,800-run results.

## Research purpose

Address two remaining reviewer concerns with a controlled extension:

1. Test whether the observed reliability-weighting result persists in larger deterministic grids with different start–goal shortest-path lengths.
2. Add tabular prioritized-sweeping Dyna-Q as a stronger planning baseline.

This extension does **not** include stochastic transitions. The current reliability estimator scores exact next-state agreement, so a stochastic experiment should be designed separately using a probability-aware predictive model and scoring rule.

## Environments

| Environment | Grid | Free states | Shortest-path length |
|---|---:|---:|---:|
| `large_12x12_short` | 12 × 12 | 121 | 22 moves |
| `large_12x12_long` | 12 × 12 | 102 | 30 moves |
| `large_16x16_medium` | 16 × 16 | 203 | 25 moves |
| `large_16x16_long` | 16 × 16 | 172 | 40 moves |

Layouts are static and deterministic in the code, so all participants run on the exact same maps. The path lengths are intentionally different from the original benchmark's 14-move shortest path.

## Algorithms

- `q_learning`: model-free reference; real-transition updates only.
- `dyna_q_fixed`: uniform random model-pair planning with fixed epsilon.
- `dyna_q_adaptive`: fixed Dyna-Q with the existing Q-dispersion epsilon heuristic.
- `dyna_q_reliability`: reliability-weighted simulated updates.
- `dyna_q_no_weight`: same planning setup but every planning update has weight 1.
- `dyna_q_shuffled_reliability`: control using reliability estimates assigned to other observed pairs.
- `dyna_q_prioritized`: prioritized sweeping with a max-priority queue, stale-entry generation checks, and the same planning-only corruption intervention. It performs up to the configured planning-update budget per real step; when the queue is empty, fewer updates may be available.

The controlled bias intervention applies only to planning predictions. Real-transition Q-updates always use the observed transition with weight 1. The reliability estimator receives prior prediction–observation records, not the corruption mask or true transition function.

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke test (pipeline verification only)

```bash
python run_scale_ps_validation.py --seeds 1 --episodes 6 --max-steps 80 --planning-steps 2 --eval-episodes 2 --bias-levels 0,0.30 --bootstrap-resamples 100 --output smoke_scale_ps
```

This configuration completes 56 small runs. It verifies execution and artifact generation; it is **not** suitable for scientific interpretation.

## Full experiment

```bash
python run_scale_ps_validation.py --seeds 10 --episodes 200 --max-steps 250 --planning-steps 5 --eval-episodes 10 --bias-levels 0,0.10,0.30 --bootstrap-resamples 30000 --output results_scale_ps
```

The full design contains 4 layouts × 3 bias levels × 7 algorithms × 10 seeds = **840 runs**. Run time depends on hardware and how often episodes reach the 250-step limit; plan for a substantial run. The program prints progress and writes results incrementally at the end of the experiment, while run artifacts are stored under the output's `artifacts/` directory.

## Saved outputs

- `config.json`: run parameters, environment layouts, paths, algorithms, seed list, and interpretive guardrails.
- `per_episode_results.csv`: one row per training episode, including steps, success, extrinsic return, and planning-update counts.
- `greedy_evaluation_results.csv`: individual greedy evaluation episodes.
- `seed_level_summary.csv`: seed-level aggregates.
- `summary_by_maze.csv`: per-layout mean and standard deviation.
- `pooled_summary.csv`: mean and standard deviation across maze–seed units at each bias level.
- `paired_bootstrap_vs_reliability.csv`: paired 95% percentile bootstrap intervals for evaluation-success differences between reliability-aware Dyna-Q and each reference/control, resampling matched maze–seed units.
- PNG figures for evaluation success, final-window training steps, and training success.
- `artifacts/<maze>/bias_<level>/<algorithm>/seed_<seed>/`: Q-table, learned-model record, prediction audit, and run metadata.
- `run_complete.json`: completion status and run duration.

## Statistical unit and interpretation

The paired bootstrap unit is one `maze × seed` combination; each unit's evaluation-success rate averages the configured evaluation episodes. The same bootstrap resample indices are not shared across comparisons, but each comparison is paired by its underlying maze–seed units. Confidence intervals are conditional on these four fixed environment layouts and should not be interpreted as confidence intervals over all possible environments. This design does not replace a stochastic benchmark or a broad state-space scaling study.

Do not merge these new data into the earlier results. Analyze this as a separate scale-and-planning-baseline extension and report outcomes whether positive, null, or negative.
