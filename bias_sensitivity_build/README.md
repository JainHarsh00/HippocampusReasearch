# Model-bias sensitivity experiment

This experiment tests how Dyna-style planning performance changes as the fraction of deliberately biased internal-model state-action pairs increases, across three fixed 8x8 mazes.

## Conditions
- Bias levels: `0, 0.05, 0.10, 0.20, 0.30` (default)
- Algorithms: model-free Q-learning, fixed-epsilon Dyna-Q, adaptive-epsilon Dyna-Q
- 3 mazes × 5 levels × 3 algorithms × 10 seeds = 450 runs by default
- The same predefined, nested corruption mask is used across levels within each maze and seed; masks are independent of algorithm visitation.
- Corruption affects simulated planning next states only. Real environment transitions and real Q-updates use observed outcomes.
- Terminal transitions are not corrupted because their next state does not affect the terminal update.

## Setup
```bash
pip install -r requirements.txt
```

## Smoke test
```bash
python run_bias_sensitivity.py --seeds 2 --episodes 20 --planning-steps 5 --eval-episodes 3 --bias-levels 0,0.1,0.3
```

## Full experiment
```bash
python run_bias_sensitivity.py --seeds 10 --episodes 200 --planning-steps 5 --eval-episodes 10 --bias-levels 0,0.05,0.10,0.20,0.30
```

Each run creates a timestamped folder under `results/`. It saves the configuration, episode and evaluation CSVs, per-seed summaries, summaries by maze, pooled descriptive summaries, plots, Q-tables, planning models, and per-run metadata.

## Interpretation guardrails
- Pooled results treat each maze-seed combination as a descriptive unit; they are not a substitute for a hierarchical statistical model.
- Compare algorithms at each bias level and inspect maze-specific results, not only pooled averages.
- The adaptive schedule is a Q-dispersion heuristic, not calibrated epistemic uncertainty.
- Results from these three mazes do not establish broad generality or a neuroscience mechanism.
- Keep this run separate from prior experiment outputs. Do not merge episode rows across implementation versions.
