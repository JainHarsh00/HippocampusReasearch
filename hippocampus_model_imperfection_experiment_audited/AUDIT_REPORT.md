# Implementation audit and corrections

This package is the audited version of the first model-imperfection experiment. It changes the implementation and therefore requires a fresh run; do not merge its outputs with the earlier `results(3).zip` results.

## Problems identified in the first version

1. The same random generator was used for agent choices/planning and environment stochasticity. Planning draws could therefore alter later environmental transitions.
2. Training and greedy evaluation shared the same environment instance and random stream.
3. Incomplete/corrupted pairs were chosen as the agent encountered them. Different algorithms could therefore receive different impairment masks because they visited different pairs.
4. Terminal transitions could be marked as corrupted even though their next-state value does not affect a terminal Q-learning target.
5. The incomplete-model condition skipped planning updates without recording how many were skipped.

## Corrections in this version

- Separate seeded random streams for agent choices/planning, training environment transitions, and evaluation environment transitions.
- A separate evaluation environment for each run.
- Impairment masks precomputed per seed over valid state-action pairs, independently of algorithm visitation. Thus algorithms with the same seed face the same preselected pair mask in a given impairment condition.
- Next-state corruption is applied only to nonterminal planning outcomes; the corrupted prediction is a valid non-goal state different from the deterministic predicted successor for that pair.
- Per-episode records distinguish attempted, applied, and skipped planning updates.
- Run metadata records the number of eligible, predefined, and observed impaired pairs.

## What is still limited

- This remains a single-maze diagnostic with tabular Q-learning and Dyna-Q.
- The biased model is a synthetic intervention, not a learned neural or hippocampal model.
- The stochastic condition is environmental action noise, not internal-model corruption.
- The incomplete-model condition omits planning updates for selected pairs; it also reduces the number of effective planning updates. Interpret it as a missing-planning-access intervention, not as an equal-compute comparison.
- The adaptive epsilon rule remains a heuristic based on Q-value dispersion, not a calibrated uncertainty estimate.
- Use independent seeds as the replication unit; episodes within a seed are not independent replicates.

## Commands

```bash
pip install -r requirements.txt
python run_model_imperfection.py --seeds 2 --episodes 20 --planning-steps 5 --eval-episodes 3
python run_model_imperfection.py --seeds 10 --episodes 200 --planning-steps 5 --eval-episodes 10
```

All outputs are saved in a new timestamped directory. Keep the original run separately for auditability.
