# Leakage-audited Reliability-Aware Dyna-Q

This is a rebuilt implementation, not a continuation of earlier results.

## Reliability protocol
1. At a state-action pair, generate a prediction from the empirical model learned from past transitions.
2. Save that prediction before stepping the environment.
3. Observe the real transition.
4. Score the saved prediction against the observed next state.
5. Update reliability only from these prior prediction/observation pairs.
6. Use reliability for subsequent planning updates.

The reliability estimator receives neither the true transition function nor the corruption-mask labels. The mask is used only by the controlled planning-model corruption intervention. Unseen pairs have no prediction and provide no reliability evidence; their reliability weight is zero until evidence exists.

## Algorithms
- `q_learning`
- `dyna_q_fixed`
- `dyna_q_adaptive`
- `dyna_q_reliability`
- `dyna_q_no_weight`
- `dyna_q_shuffled_reliability`

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

The full configuration plans 900 runs. Outputs are saved under a timestamped folder and include raw CSVs, summaries, plots, Q-tables, learned model records, prediction audits, and metadata.
