# Pre-run audit checklist

- [x] Reliability predictions are generated from learned transition outcomes only.
- [x] Prediction is saved before the current environment transition is observed.
- [x] Reliability score is updated only after a prior prediction is compared with the observation.
- [x] Reliability estimator is not passed the corruption mask or true transition function.
- [x] Unseen pairs have no reliability evidence; no evidence yields zero planning weight.
- [x] Prediction audit is saved for each reliability-based run.
- [ ] After running, inspect prediction audit and compare algorithms' outputs.
- [ ] Confirm smoke test metrics and evaluation policies before launching full run.

Note: the corruption intervention itself intentionally alters planning predictions for
predefined state-action pairs. The estimator does not get the mask labels, but it can
infer unreliability through prediction errors after real observations. This is an
experimental heuristic, not a calibrated uncertainty estimator.
