# Protocol: Model-bias sensitivity

## Research question
How does a controlled increase in biased simulated next-state predictions affect model-free and model-based RL performance, and does adaptive epsilon change the sensitivity?

## Independent variables
1. Bias level: 0%, 5%, 10%, 20%, 30% of eligible nonterminal state-action pairs.
2. Algorithm: Q-learning, fixed-epsilon Dyna-Q, adaptive-epsilon Dyna-Q.
3. Maze: three fixed layouts.

## Outcomes
- Primary: greedy evaluation success rate.
- Secondary: final-window training success, final-window episode steps, extrinsic return, and successful evaluation path length.

## Pairing and reproducibility
For each maze and seed, one shuffled list of eligible pairs determines nested masks at all bias levels. The mask is independent of visitation and algorithm. Agent, training environment, evaluation environment, and greedy action streams are seeded separately. Q-learning is repeated at every nominal bias level as a negative-control series; its learning rule does not use the model.

## Analysis plan
1. Validate run completion, configuration, and expected row counts.
2. Plot evaluation success against bias level by algorithm and inspect per-maze curves.
3. Compare algorithms within the same maze/seed/level. Treat seed-level runs—not episodes—as the independent replicates within each maze.
4. Report means, standard deviations, and confidence intervals or paired tests only after checking distributions and paired differences.
5. Avoid claims that adaptive exploration solves model bias unless the result is consistent across mazes and robust statistical comparisons.

## Known limitations
The three mazes are small, handcrafted, and deterministic at the environment level. The biased planning model substitutes a fixed incorrect next state for selected pairs; it does not represent all forms of learned-model error. The adaptive epsilon heuristic is not a calibrated uncertainty measure.
