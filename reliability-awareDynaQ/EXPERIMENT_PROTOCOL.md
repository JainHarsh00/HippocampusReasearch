# Experimental Protocol: Reliability-Aware Dyna-Q

## Research question
Can an online estimate of model prediction reliability reduce the performance degradation caused by biased simulated transitions in Dyna-Q?

## Conditions
Three fixed mazes; predefined nested model-bias masks at 0%, 5%, 10%, 20%, and 30%; four algorithms; 10 seeds. The full design contains 600 algorithm-condition runs.

## Primary outcome
Greedy evaluation success rate, averaged over 10 evaluation episodes per seed. Secondary outcomes: final-window training success, final-window episode length, extrinsic return, and average reliability weight applied to planning updates.

## Method
Reliability is estimated separately for each state-action pair from agreement between the model's planning prediction and observed real next states. Laplace smoothing is used: `(matches + 1) / (observations + 2)`. For the reliability-aware algorithm, this estimate scales the learning rate of planning updates only. Real experience updates remain unchanged.

## Comparators
Model-free Q-learning; fixed-epsilon Dyna-Q; the existing adaptive-epsilon Dyna-Q heuristic; reliability-aware Dyna-Q.

## Reproducibility
Random streams for the agent, training environment, evaluation environment, and greedy action selection are separate. Bias masks are pre-sampled per maze and seed, nested across bias levels, and independent of algorithm visitation. All settings and artifacts are saved.

## Analysis plan
Report results by maze and pooled over maze-seed units. For paired contrasts, match the same maze, seed, and bias level. Report effect sizes and confidence intervals, and correct for multiple comparisons when conducting multiple tests. Do not treat episodes within a run as independent samples. Do not tune on evaluation mazes.

## Limitations
Only three fixed handcrafted mazes are included. The reliability estimator is a simple prediction-agreement heuristic; it may respond slowly to model errors and is not a calibrated probability. The corruption mechanism is artificial and affects planning predictions only. Any positive result requires validation on held-out environments and sensitivity analysis before broad claims.
