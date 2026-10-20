# Smoke-test report

Smoke tests completed locally on 2026-10-18 with Python and CPU PyTorch.

## Intrinsic-baseline script
Command:

```bash
python run_intrinsic_baselines.py --seeds 1 --episodes 3 --max-steps 12 --eval-episodes 2 --slip-levels 0.10 --curiosity-train-every 2 --output smoke_intrinsic_final
```

Result: 10/10 short configurations finished (two maps × five algorithms). Per-episode/evaluation CSVs, seed/pooled summaries, Q-tables, intrinsic module snapshots, configuration, and per-run metadata were generated.

## UCT-MCTS script
Command:

```bash
python run_mcts_baseline.py --seeds 1 --episodes 3 --max-steps 12 --eval-episodes 2 --slip-levels 0.10 --bias-levels 0,0.30 --mcts-simulations 4 --search-depth 4 --output smoke_mcts_final
```

Result: 4/4 short configurations finished (two maps × two corruption levels). Episode/evaluation records, seed/pooled summaries, Q-tables, learned model snapshots, configuration, and run metadata were generated.

These are software smoke tests only, not scientifically meaningful performance experiments. Do not cite smoke-test performance in the manuscript.
