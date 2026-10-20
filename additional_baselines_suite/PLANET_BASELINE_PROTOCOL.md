# PlaNet baseline: separate port/reproduction required

PlaNet is deliberately not marked as implemented in this package. The primary publication models image observations with a recurrent state-space model containing deterministic and stochastic latent components, trains the model to predict observations and rewards, and plans in latent space using candidate action sequences. A tabular transition table or small MLP does not meet that definition.

If a PlaNet comparison is required for the final manuscript, use a separate development branch with the following safeguards:

1. Wrap the grid as a pixel-observation environment; define the four actions as discrete controls.
2. Preserve PlaNet's core elements (image encoder/decoder, deterministic recurrent state, stochastic latent prior/posterior, reward prediction, and latent-space model-predictive planning). The planner will need a documented discrete-action adaptation; do not call it an exact reproduction without verifying the method-level correspondence.
3. Train only from observed transitions, with no access to the true transition table for learning/planning.
4. Define a compute and data budget before the run, and report wall-clock, environment interactions, model-training updates, and planning model calls.
5. Compare across the same held-out stochastic mazes and action-slip probabilities. The original planning-only corruption intervention may not map cleanly into a learned latent model; do not silently corrupt the latent model using simulator ground truth.
6. Validate observation reconstruction, reward prediction, multi-step predictive accuracy, policy success, and reproducibility before treating performance as evidence about the full method.

Official source and paper:
- Hafner et al. (2019), *Learning Latent Dynamics for Planning from Pixels*, PMLR 97. https://proceedings.mlr.press/v97/hafner19a.html
- Official implementation: https://github.com/google-research/planet

This protocol is a plan only. PlaNet has **not** been run in the current study. An adapted port should be referred to as a “PlaNet-inspired discrete-action latent-model baseline,” not canonical PlaNet.
