# Stochastic-Transition Validation for Reliability-Weighted Dyna-Q

## Purpose

This is a separate follow-up experiment for the reviewer request to evaluate stochastic transitions. It should be run after the completed larger-grid/prioritized-sweeping experiment. It does **not** overwrite or reuse the prior results directory.

It examines whether reliability-weighted planning remains useful when the environment has stochastic action-slip dynamics and when selected state-action planning forecasts are artificially corrupted.

## Experiment design

The default full run contains:

- **2 fixed layouts:** `large_12x12_short` (shortest path 22 moves) and `large_16x16_long` (shortest path 40 moves).
- **2 stochasticity levels:** total slip probability 0.10 and 0.25. The intended action executes with probability `1 - slip`; each perpendicular action executes with probability `slip / 2`. Blocked moves leave the agent in place.
- **3 planning-model corruption levels:** 0%, 10%, and 30% of eligible state-action pairs. For a selected pair, the distribution used for planning is replaced by a nonlocal, non-goal distractor state. This intervention does not change the real environment transition or the real-transition Q-update.
- **7 algorithms:** Q-learning, fixed Dyna-Q, adaptive Dyna-Q, reliability-aware Dyna-Q, Dyna-Q without weighting, Dyna-Q with shuffled reliability, and prioritized sweeping.
- **10 seeds per exact condition.**

Total: `2 layouts × 2 slip levels × 3 corruption levels × 7 algorithms × 10 seeds = 840 runs`.

The corruption fraction applies to eligible state-action pairs, not to the probability of a stochastic slip. The mask is nested and shared across algorithms for each maze and seed. The environment's stochastic transitions remain active during both training and greedy evaluation.

## Reliability estimator

The learned transition model stores empirical categorical frequencies over next states previously observed for each state-action pair. At a real interaction, the model's *planning forecast* is computed before the environment observation is known. After observing the next state, the estimator records the forecast's capped prequential negative log-likelihood (NLL). The NLL is capped at `log(number_of_valid_states)`, the loss of a uniform-over-valid-cells reference. The planning weight is

`w(s,a) = n / (n + 2) × clip(1 - mean_NLL / log(|S_valid|), 0, 1)`,

where `n` is the number of prior forecasts scored for that pair. With no scored forecasts, the weight is zero.

This is a heuristic predictive-skill weight, **not a calibrated probability**. Unlike exact-match reliability, it evaluates the predicted distribution and can assign partial credit to a stochastic outcome when that outcome already has nonzero empirical probability. An outcome never observed before receives the capped surprise loss for that forecast; as the model learns new outcomes, later forecasts can improve.

The reliability estimator receives neither the corruption mask nor the true transition function. The intervention layer modifies planning forecasts only. Q-updates from actual environment transitions always use weight 1.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Step 1: Smoke test

Run this first; it completes 56 small configurations and checks that the output pipeline works.

```bash
python run_stochastic_reliability.py --seeds 1 --episodes 6 --max-steps 60 --planning-steps 2 --eval-episodes 2 --slip-levels 0.10,0.25 --bias-levels 0,0.30 --bootstrap-resamples 100 --output smoke_stochastic
```

A second terminal sanity check (14 configurations) can exercise longer learning episodes at zero corruption:

```bash
python run_stochastic_reliability.py --seeds 1 --episodes 40 --max-steps 250 --planning-steps 5 --eval-episodes 5 --slip-levels 0.10 --bias-levels 0 --bootstrap-resamples 100 --output smoke_success
```

Both smoke tests passed during package preparation. They are code/pipeline checks, not scientific results.

## Step 2: Full run

```bash
python run_stochastic_reliability.py --seeds 10 --episodes 200 --max-steps 250 --planning-steps 5 --eval-episodes 10 --slip-levels 0.10,0.25 --bias-levels 0,0.10,0.30 --bootstrap-resamples 30000 --output results_stochastic
```

Expected design: 840 runs. The script writes per-run episode/evaluation rows incrementally so an interruption does not lose completed aggregate rows. Each run also writes a Q-table, compact learned-model summary, reliability-audit summary, and run metadata.

## Output files

- `config.json`: exact experiment design, Python/NumPy versions, and SHA-256 of the script.
- `per_episode_results.csv`: training episode data, including model coverage and planning update counts.
- `greedy_evaluation_results.csv`: all evaluation episodes.
- `seed_level_summary.csv`, `summary_by_maze.csv`, and `pooled_summary.csv`: seed-level means, SDs, and pooled summaries.
- `paired_bootstrap_vs_reliability.csv`: paired bootstrap differences and percentile 95% confidence intervals against all other algorithms.
- `paired_tests_vs_reliability.csv`: paired Wilcoxon signed-rank tests with Holm correction across the listed comparisons.
- `evaluation_success_*.png`: per-condition evaluation plots.
- `artifacts/...`: learned Q-table, transition-count/model summary, reliability audit, and run metadata for every condition.
- `run_complete.json`: completion count, elapsed time, and source hash.

## Interpretation safeguards

1. Run-level confidence intervals and tests are conditional on these **two fixed layouts** and the selected corruption masks; they are not evidence of generalization to every stochastic environment.
2. This tests one known action-slip family, not arbitrary stochastic or nonstationary dynamics.
3. The planning-corruption mechanism is synthetic and severe: selected pairs use a nonlocal distractor successor. It should not be equated with every natural form of model error.
4. Prioritized sweeping uses expected TD error and expected backup targets under the empirical model distribution. It may perform fewer updates than the configured maximum when its priority queue has no eligible entries; check recorded update counts before making compute-budget claims.
5. Do not tune a method based on these full-run outcomes and then report the same outcomes as an untouched test. Any method change should be labeled as a new experiment.
