# Held-out Maze Validation

This experiment adds three new maze layouts (`heldout_d`, `heldout_e`, `heldout_f`) not used in earlier method development. Existing mazes remain in the run as reference conditions. All six algorithms and five model-bias levels are evaluated with the same protocol.

Important: these are held-out layouts from the prior experiments, but they are still handcrafted, fixed mazes. This validates transfer to new layouts, not generality across arbitrary environments. Keep these results separate from all prior runs.

## Install
```bash
pip install -r requirements.txt
```

## Smoke test
```bash
python run_heldout_validation.py --seeds 2 --episodes 20 --planning-steps 5 --eval-episodes 3 --bias-levels 0,0.1,0.3
```

## Full validation
```bash
python run_heldout_validation.py --seeds 10 --episodes 200 --planning-steps 5 --eval-episodes 10 --bias-levels 0,0.05,0.10,0.20,0.30
```

The full run has 6 mazes × 5 bias levels × 6 algorithms × 10 seeds = 1,800 runs.
