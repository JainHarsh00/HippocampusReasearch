# Audit of the 18 October baseline results

The archive `results(20261018-103359).zip` contains a completed intrinsic-baseline track (200/200 runs) and a completed MCTS track (120/120 runs). Episode/evaluation record counts and summary means were verified from raw CSV files; duplicate episode/evaluation keys were not found. Recorded source hashes match the scripts in the original baseline package.

## Intrinsic-exploration track

The evaluated agents are tabular Q-learning, count-bonus Q-learning, and discrete-state adaptations labelled RND-style, ICM-style, and NGU-style. They are not exact reproductions of the original deep-RL algorithms.

On the 12x12 maze, greedy evaluation success was (10% slip / 25% slip): Q-learning 62% / 73%; count bonus 25% / 71%; RND-style 37% / 76%; ICM-style 8% / 0%; NGU-style 61% / 67%. All methods achieved 0% on the 16x16 maze (40-move shortest path) under the current 200-episode budget. The pooled results therefore combine one partly learnable task with one task where no baseline succeeds; do not rank these methods solely from the pooled means.

## MCTS track

The initial MCTS run used 16 simulations and search depth 12, while shortest-path lengths were 22 and 40 moves. Because the root search horizon was shorter than either shortest path, the tree search could not see terminal reward from the start state on a shortest route. On the 12x12 maze, evaluation success was only 16% / 32% at zero corruption for slip 10% / 25%; the 16x16 maze had 0% success in every condition. The run is complete as a data-processing matter, but is not a well-powered MCTS comparison.

The initial summary metric named `mean_mcts_simulations_per_training_step` actually averaged the total simulations summed within each episode, so the label is inaccurate. The revised script reports simulation counts per episode and per environment step separately.

## Required interpretation

Treat these as pilot results. Do not claim canonical RND/ICM/NGU reproduction. Do not use the initial MCTS scores as evidence that MCTS is inferior; rerun after increasing the search depth beyond 40 and increasing the simulation budget. A 48-depth, 32-simulation default is provided in the updated script. Even that result should be described as a UCT-MCTS plus Q-bootstrap hybrid, with compute budget reported. The long maze still requires a better training protocol or more training budget if it is to serve as a meaningful comparison rather than a universal failure case.
