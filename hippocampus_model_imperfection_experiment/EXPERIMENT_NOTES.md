# Experiment protocol and preregistered interpretation

## Research question
Does Dyna-style simulated planning become less reliable when the internal transition model is incomplete or biased, and does the tested adaptive-epsilon heuristic improve robustness relative to fixed epsilon?

## Design
A 4-condition × 3-algorithm design, with independent random seeds within each condition:

| Condition | Manipulation |
|---|---|
| Accurate deterministic | No environment or planning-model impairment |
| Stochastic transitions | Random executed action with configured slip probability |
| Incomplete model | Fixed subset of observed state-action pairs excluded from planning |
| Biased model | Fixed subset of planning predictions use an incorrect next state |

Algorithms: model-free Q-learning, fixed-epsilon Dyna-Q, adaptive-epsilon Dyna-Q. Environment rewards and real-interaction updates are held constant across algorithms. Planning-model impairment is applied only to planning updates.

## Primary outcomes
- Greedy evaluation success rate per seed
- Mean steps in successful greedy evaluation episodes per seed
- Final-20%-of-training success rate and steps per seed
- Extrinsic return (not intrinsic reward)

## Analysis rule
Summarize across independent seeds using mean and standard deviation. Do not treat each episode as an independent replicate. Compare adaptive vs fixed Dyna-Q within each condition; emphasize effect sizes and uncertainty. Avoid significance claims until a pre-specified inferential analysis is performed.

## Limitations
One maze, one impairment strength per default run, one fixed epsilon, and one adaptive schedule setting. If results look promising, follow with multiple held-out mazes, impairment-strength sweeps, and a separate tuning/evaluation split. This experiment alone cannot establish biological validity or general adaptive-exploration superiority.
