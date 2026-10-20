# Reliability-Aware Dyna-Q — Ablation Experiment

Six algorithms are compared:
- `q_learning`: model-free baseline
- `dyna_q_fixed`: fixed-epsilon Dyna-Q
- `dyna_q_adaptive`: adaptive-epsilon Dyna-Q
- `dyna_q_reliability`: pair-aligned reliability-weighted planning
- `dyna_q_no_weight`: same model and reliability tracking, but unweighted planning updates
- `dyna_q_shuffled_reliability`: reliability scores permuted across experienced pairs

The estimator uses learned-model predictions and subsequent observations; it does not
query the true transition function or receive corruption labels. Keep this run separate
from earlier implementations.

## Install
```bash
pip install -r requirements.txt
```

## Smoke test
```bash
python run_reliability_aware.py --seeds 2 --episodes 20 --planning-steps 5 --eval-episodes 3 --bias-levels 0,0.1,0.3
```

## Full run
```bash
python run_reliability_aware.py --seeds 10 --episodes 200 --planning-steps 5 --eval-episodes 10 --bias-levels 0,0.05,0.10,0.20,0.30
```

The full run contains 900 runs.
