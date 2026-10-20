# Baseline coverage and claim guardrails

| Requested method | Included in this package? | What the code actually tests | Claim allowed in paper |
|---|---|---|---|
| RND | Yes, Track A | Fixed random target network + learned predictor over discrete state one-hots; normalized prediction error bonus | “RND-style discrete-state adaptation”; not a direct reproduction of Atari RND |
| ICM | Yes, Track A | Learned feature encoder, inverse action prediction, forward feature prediction error | “ICM-style discrete-state adaptation”; not a full image-based ICM reproduction |
| NGU | Partial adaptation, Track A | Episodic novelty + RND lifelong novelty | “NGU-style dual-timescale novelty adaptation”; not canonical NGU because no UVFA policy family or distributed recurrent training |
| MCTS | Yes, Track B | UCT over learned stochastic transition model; Q-learning real updates and Q-value bootstrap leaves | “UCT-MCTS + Q-bootstrap hybrid”; disclose tree budget and computational mismatch |
| PlaNet | No | No code/run is included | Do not claim PlaNet was tested; a separate RSSM/latent-planning adaptation is needed |

All results must be run and audited before being added to the paper. Do not report the smoke-test outputs as experimental findings.
