# Additional baselines for the hippocampal-planning manuscript

This package adds two **separate** experimental tracks on the same stochastic grid worlds used by the existing reliability-weighting study. It preserves the existing experiment archives and writes to new output folders.

## What is included

### Track A: intrinsic-exploration baselines

`run_intrinsic_baselines.py` compares:

- `q_learning`: tabular Q-learning, extrinsic reward only.
- `count_bonus_q`: tabular Q-learning with a count-based intrinsic reward.
- `rnd_q`: a discrete-state RND-style variant (fixed random target MLP; predictor trained to match its target features).
- `icm_q`: a discrete-state ICM-style variant (learned feature encoder, inverse action predictor, and forward feature predictor; forward feature error supplies intrinsic reward).
- `ngu_style_q`: a discrete-state NGU-style dual-timescale reward (episodic novelty in a fixed random embedding multiplied by lifelong RND novelty).

These methods are adaptations for small, fully observed state spaces, not exact reproductions of the original high-dimensional deep-RL systems. In particular, `ngu_style_q` does **not** implement NGU's original distributed recurrent agent or family of policies trained using universal value-function approximation. The paper must label these variants as discrete-state adaptations, not claim a direct reproduction of canonical RND, ICM, or NGU.

This track has no injected model-corruption factor, because these variants are model-free exploration agents: the planning-only corruption intervention does not affect their update mechanism. It isolates whether intrinsic exploration improves stochastic-navigation learning. The extrinsic reward is always used for reported return/evaluation.

Quick smoke test:

```bash
python run_intrinsic_baselines.py --seeds 1 --episodes 3 --max-steps 12 --eval-episodes 2 --slip-levels 0.10 --curiosity-train-every 2 --output smoke_intrinsic
```

Full run (200 runs: 2 mazes × 2 slip settings × 5 algorithms × 10 seeds):

```bash
python run_intrinsic_baselines.py --seeds 10 --episodes 200 --max-steps 250 --eval-episodes 10 --slip-levels 0.10,0.25 --intrinsic-scale 0.05 --curiosity-train-every 4 --output results_intrinsic
```

### Track B: UCT-MCTS planning baseline

`run_mcts_baseline.py` adds a UCT tree-search agent using an online empirical transition model. It shares the study's planning-only corruption intervention. Real transitions update a tabular Q-value table; Q values bootstrap leaf estimates when a tree rollout hits an unknown pair or the depth limit. This is an explicit **UCT-MCTS + Q-value bootstrap hybrid**, not a claim to reproduce every canonical MCTS-RL algorithm.

The search performs a fixed number of simulations per real decision and has a depth cap. Its model-query/computational budget is larger than the five one-step Dyna updates in the existing study, so report the budget and avoid calling this a compute-matched comparison.

Quick smoke test:

```bash
python run_mcts_baseline.py --seeds 1 --episodes 3 --max-steps 12 --eval-episodes 2 --slip-levels 0.10 --bias-levels 0,0.30 --mcts-simulations 4 --search-depth 4 --output smoke_mcts
```

Full run (120 runs: 2 mazes × 2 slip settings × 3 corruption levels × 10 seeds):

```bash
python run_mcts_baseline.py --seeds 10 --episodes 200 --max-steps 250 --eval-episodes 10 --slip-levels 0.10,0.25 --bias-levels 0,0.10,0.30 --mcts-simulations 16 --search-depth 12 --uct-exploration 1.414 --output results_mcts
```

The existing stochastic experiment source is bundled as `run_stochastic_reliability.py` and imported only for shared environment, corruption-design, and transition-distribution helpers. Both scripts record the source hash, helper hash, Python/Torch/NumPy versions as applicable, and save results incrementally after each completed run.

## PlaNet: separate implementation track required

PlaNet is not included in this package because a faithful comparison requires a learned recurrent stochastic latent dynamics model, observation encoder/decoder, and latent-space planning. The published method is built around learning latent dynamics for planning from images; a tabular Q-value baseline or a small MLP alone would not be PlaNet. A discrete-action maze adaptation is possible, but it must be labelled as an adaptation and given its own training budget, image-observation interface, validation checks, and compute accounting.

Do not write that PlaNet was tested until that separate implementation has actually run and passed an audit.

## Output artifacts

Each track saves:

- `config.json` and `run_complete.json`
- `per_episode_results.csv`
- `greedy_evaluation_results.csv`
- `seed_level_summary.csv` and `pooled_summary.csv`
- per-run Q-tables, model/module snapshots, and metadata under `artifacts/`

Results from the two tracks should remain separate and be analysed against the matching reference conditions in the previously completed stochastic experiment. Do not combine evaluation episodes as if they were independent random seeds; use maze-seed units for paired comparisons.

## References for method definitions

- RND: Burda et al., *Exploration by Random Network Distillation*. [arXiv:1810.12894](https://arxiv.org/abs/1810.12894); [OpenAI reference implementation](https://github.com/openai/random-network-distillation).
- ICM: Pathak et al., *Curiosity-driven Exploration by Self-supervised Prediction*. ICML 2017, PMLR 70. [Paper](https://proceedings.mlr.press/v70/pathak17a.html).
- NGU: Badia et al., *Never Give Up: Learning Directed Exploration Strategies*. ICLR 2020. [Paper entry](https://mlanthology.org/iclr/2020/badia2020iclr-never/).
- PlaNet: Hafner et al., *Learning Latent Dynamics for Planning from Pixels*. ICML 2019, PMLR 97. [Paper](https://proceedings.mlr.press/v97/hafner19a.html); [official code](https://github.com/google-research/planet).
- MCTS overview: Świechowski et al., *Monte Carlo Tree Search: a review of recent modifications and applications*. Artificial Intelligence Review 56 (2023). [Article](https://doi.org/10.1007/s10462-022-10228-y).
