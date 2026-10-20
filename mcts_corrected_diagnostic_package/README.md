# MCTS Hybrid Diagnostic — Corrected Root Selection

## Purpose
This is a diagnostic, not a publication-ready MCTS benchmark. It tests whether selecting the root action by estimated mean return (visit count only as a tie-break) fixes the observed failure on an open 8×8 grid. It also adds explicit epsilon-greedy real-action exploration.

## Findings
- Original visit-count-first root selection with epsilon=0.2, 3 seeds × 60 episodes: tail training success 8.3%; greedy evaluation success 0%.
- Return-first root selection with epsilon=0.2, 3 seeds × 60 episodes: tail training success 69.4%; greedy evaluation success 53.3%.
- Return-first root selection with epsilon=0.2, 5 seeds × 100 episodes: tail training success 99%; greedy evaluation success 100%; mean evaluation length 40.22 steps.

The first two comparisons use the same diagnostic budget (8 simulations, depth 12), but are small pilot runs. The 5-seed run is stronger evidence that the corrected action-selection variant can learn the simple environment, not evidence that it is a canonical or compute-matched MCTS baseline.

## Important caveats
- The method remains a hybrid: UCT over an online empirical model, tabular Q-learning on real transitions, and Q-value bootstrapping at unknown search leaves.
- Explicit epsilon exploration is an additional intervention and must be reported.
- The open grid has a 14-move shortest path; these tests do not validate performance on the original 12×12/16×16 mazes, stochastic conditions, or model corruption.
- Do not cite these diagnostic results as a final comparative result. Next validate on the original maze with zero slip/bias, then test multiple seeds and compare compute budgets.

## Run
Run from a Python environment with NumPy and the existing `run_stochastic_reliability.py` and `run_mcts_baseline.py` files from the corrected MCTS suite in the same directory. The diagnostic script imports those helpers. The summary JSONs record the source hash and runtime versions.
