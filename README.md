# Adaptive-epsilon diagnostic sweep

This diagnostic compares fixed-epsilon Dyna-Q (epsilon=0.2) against adaptive-epsilon Dyna-Q over a predefined lambda grid. Curiosity is intentionally disabled to isolate the adaptive exploration mechanism.

## Install
```bash
pip install -r requirements.txt
```

## Quick smoke test
```bash
python diagnose_adaptive_epsilon.py --seeds 2 --episodes 20 --lambdas 0,1,5
```

## Main diagnostic
```bash
python diagnose_adaptive_epsilon.py --seeds 10 --episodes 200 --planning-steps 5 --lambdas 0,0.5,1,2,5,10
```

Run from this folder, which must contain both `diagnose_adaptive_epsilon.py` and `run_experiments.py`.

## What is controlled?
- Same maze, learning rate, discount factor, epsilon bounds, planning budget, episode cap, and matched seed numbers.
- Fixed baseline uses epsilon=0.2.
- Adaptive conditions differ only in lambda.
- Curiosity is disabled in all conditions.

## Files saved
Every run gets a new timestamped folder under `results/`, containing config, per-episode results, evaluation results, per-seed summaries, across-seed summary, plots, Q tables, learned models, and metadata.

## Important
This is diagnostic tuning, not confirmatory evaluation. Do not select lambda solely on the final evaluation set and then report that same set as unbiased test performance. After choosing a candidate using training/validation criteria, freeze it and evaluate it on held-out seeds and additional mazes.

The adaptive rule is still a heuristic based on action-value dispersion; the experiment tests its behavior, not whether dispersion is a calibrated epistemic uncertainty estimate.
