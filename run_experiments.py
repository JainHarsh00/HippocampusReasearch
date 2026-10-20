#!/usr/bin/env python3
"""
Reproducible tabular RL baseline for the hippocampal-inspired planning project.

Algorithms:
  q_learning_fixed
  dyna_q_fixed
  dyna_q_curiosity
  dyna_q_adaptive
  dyna_q_curiosity_adaptive

The adaptive epsilon rule is deliberately treated as a heuristic, not as a
validated epistemic-uncertainty estimator:
  epsilon(s) = eps_min + (eps_max - eps_min) * exp(-lambda * range_a Q(s,a))

This initial experiment uses one deterministic maze only. It is a pipeline
smoke test and baseline, not sufficient evidence for the paper's general claims.
All random seeds, configurations, per-episode metrics, summary statistics,
Q tables, learned transition models, and plots are saved to a new run folder.
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import random
import statistics
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ACTIONS = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1),  # left
}
ACTION_NAMES = ["up", "right", "down", "left"]

# Fixed 8x8 maze. '#' = wall, S = start, G = goal.
MAZE = [
    "S...#...",
    ".##.#.#.",
    "...#....",
    ".#...##.",
    ".###.#..",
    "...#.#..",
    "##...#..",
    ".......G",
]
ROWS = len(MAZE)
COLS = len(MAZE[0])
START = next((r, c) for r, row in enumerate(MAZE) for c, ch in enumerate(row) if ch == "S")
GOAL = next((r, c) for r, row in enumerate(MAZE) for c, ch in enumerate(row) if ch == "G")
WALLS = {(r, c) for r, row in enumerate(MAZE) for c, ch in enumerate(row) if ch == "#"}


def shortest_path_length():
    q = deque([(START, 0)])
    seen = {START}
    while q:
        (r, c), d = q.popleft()
        if (r, c) == GOAL:
            return d
        for dr, dc in ACTIONS.values():
            nxt = (r + dr, c + dc)
            nr, nc = nxt
            if 0 <= nr < ROWS and 0 <= nc < COLS and nxt not in WALLS and nxt not in seen:
                seen.add(nxt)
                q.append((nxt, d + 1))
    raise RuntimeError("Maze is unsolvable. Fix MAZE before running experiments.")


class GridWorld:
    def __init__(self, step_penalty=-0.01, goal_reward=1.0):
        self.step_penalty = float(step_penalty)
        self.goal_reward = float(goal_reward)
        self.state = START

    def reset(self):
        self.state = START
        return self.state

    def step(self, action):
        r, c = self.state
        dr, dc = ACTIONS[int(action)]
        candidate = (r + dr, c + dc)
        nr, nc = candidate
        if not (0 <= nr < ROWS and 0 <= nc < COLS) or candidate in WALLS:
            candidate = self.state
        self.state = candidate
        done = candidate == GOAL
        reward = self.goal_reward if done else self.step_penalty
        return candidate, reward, done


def epsilon_for_state(q_values, cfg, adaptive):
    if not adaptive:
        return cfg["epsilon_fixed"]
    dispersion = float(np.max(q_values) - np.min(q_values))
    # Heuristic: equal action values -> high exploration; separated values -> less.
    # This is NOT a calibrated epistemic-uncertainty estimate.
    raw = cfg["epsilon_min"] + (cfg["epsilon_max"] - cfg["epsilon_min"]) * math.exp(
        -cfg["lambda_dispersion"] * dispersion
    )
    return float(np.clip(raw, cfg["epsilon_min"], cfg["epsilon_max"]))


def select_action(q_table, state, rng, cfg, adaptive):
    eps = epsilon_for_state(q_table[state], cfg, adaptive)
    if rng.random() < eps:
        return rng.randrange(len(ACTIONS)), eps
    values = q_table[state]
    best = np.flatnonzero(np.isclose(values, np.max(values)))
    return int(rng.choice(best.tolist())), eps


def td_update(q_table, state, action, reward, next_state, done, cfg):
    bootstrap = 0.0 if done else float(np.max(q_table[next_state]))
    target = reward + cfg["gamma"] * bootstrap
    q_table[state, action] += cfg["alpha"] * (target - q_table[state, action])


def run_one(seed, algorithm, cfg, output_dir):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    env = GridWorld(cfg["step_penalty"], cfg["goal_reward"])
    n_states = ROWS * COLS
    q = np.zeros((n_states, len(ACTIONS)), dtype=np.float64)
    counts = np.zeros((n_states, len(ACTIONS)), dtype=np.int64)

    use_planning = algorithm != "q_learning_fixed"
    use_curiosity = "curiosity" in algorithm
    adaptive = "adaptive" in algorithm
    # Model stores (next_state_id, extrinsic_reward, done) for observed pairs.
    model = {}
    experienced_pairs = []
    visited_pairs = set()

    def sid(state):
        return state[0] * COLS + state[1]

    episode_rows = []
    for episode in range(1, cfg["episodes"] + 1):
        state = env.reset()
        steps = 0
        ext_return = 0.0
        success = False
        eps_values = []
        planning_updates = 0

        while steps < cfg["max_steps_per_episode"]:
            state_id = sid(state)
            action, eps = select_action(q, state_id, rng, cfg, adaptive)
            eps_values.append(eps)
            next_state, ext_reward, done = env.step(action)
            next_id = sid(next_state)

            counts[state_id, action] += 1
            intrinsic = (
                cfg["curiosity_beta"] / math.sqrt(1.0 + counts[state_id, action])
                if use_curiosity else 0.0
            )
            learning_reward = ext_reward + intrinsic
            td_update(q, state_id, action, learning_reward, next_id, done, cfg)

            pair = (state_id, action)
            model[pair] = (next_id, float(ext_reward), bool(done))
            if pair not in visited_pairs:
                visited_pairs.add(pair)
                experienced_pairs.append(pair)

            if use_planning and experienced_pairs:
                for _ in range(cfg["planning_steps"]):
                    ps, pa = rng.choice(experienced_pairs)
                    pnext, pext, pdone = model[(ps, pa)]
                    pintrinsic = (
                        cfg["curiosity_beta"] / math.sqrt(1.0 + counts[ps, pa])
                        if use_curiosity else 0.0
                    )
                    td_update(q, ps, pa, pext + pintrinsic, pnext, pdone, cfg)
                    planning_updates += 1

            ext_return += ext_reward
            steps += 1
            state = next_state
            if done:
                success = True
                break

        episode_rows.append({
            "algorithm": algorithm,
            "seed": seed,
            "episode": episode,
            "steps": steps,
            "success": int(success),
            "extrinsic_return": ext_return,
            "mean_epsilon": float(np.mean(eps_values)) if eps_values else None,
            "planning_updates": planning_updates,
            "unique_state_action_pairs": len(visited_pairs),
        })

    # Greedy evaluation: no exploration; does not update Q, counts, or model.
    eval_rows = []
    for eval_episode in range(1, cfg["evaluation_episodes"] + 1):
        state = env.reset()
        ext_return = 0.0
        success = False
        steps = 0
        visited = set()
        while steps < cfg["max_steps_per_episode"]:
            state_id = sid(state)
            values = q[state_id]
            best = np.flatnonzero(np.isclose(values, np.max(values)))
            action = int(np_rng.choice(best.tolist()))
            next_state, reward, done = env.step(action)
            ext_return += reward
            steps += 1
            state = next_state
            if done:
                success = True
                break
            # Prevent endless greedy cycles from consuming the entire budget.
            if state in visited:
                # Keep the episode running until max steps so the cap is explicit.
                pass
            visited.add(state)
        eval_rows.append({
            "algorithm": algorithm,
            "seed": seed,
            "evaluation_episode": eval_episode,
            "steps": steps,
            "success": int(success),
            "extrinsic_return": ext_return,
        })

    seed_dir = Path(output_dir) / "artifacts" / algorithm / f"seed_{seed:04d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(seed_dir / "q_table.npz", q_table=q, action_names=np.array(ACTION_NAMES))
    model_payload = []
    for (state_id, action), (next_id, reward, done) in sorted(model.items()):
        model_payload.append({
            "state": [state_id // COLS, state_id % COLS],
            "action": ACTION_NAMES[action],
            "next_state": [next_id // COLS, next_id % COLS],
            "extrinsic_reward": reward,
            "done": done,
            "real_visit_count": int(counts[state_id, action]),
        })
    with open(seed_dir / "learned_model.json", "w", encoding="utf-8") as f:
        json.dump(model_payload, f, indent=2)
    with open(seed_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "algorithm": algorithm,
            "config": cfg,
            "maze": MAZE,
            "start": START,
            "goal": GOAL,
            "shortest_path_length_moves": shortest_path_length(),
            "q_table_shape": list(q.shape),
            "learned_state_action_pairs": len(model),
        }, f, indent=2)

    return episode_rows, eval_rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    vals = [float(v) for v in values]
    if not vals:
        return None, None
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def main():
    parser = argparse.ArgumentParser(description="Run reproducible tabular RL baseline experiments.")
    parser.add_argument("--seeds", type=int, default=10, help="Number of independent random seeds (default: 10 for initial smoke-test-scale runs).")
    parser.add_argument("--episodes", type=int, default=200, help="Training episodes per algorithm and seed.")
    parser.add_argument("--max-steps", type=int, default=150, help="Maximum environment steps per episode.")
    parser.add_argument("--planning-steps", type=int, default=5, help="Dyna planning updates after each real transition.")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Greedy evaluation episodes after training.")
    parser.add_argument("--output", type=str, default="results", help="Base output directory.")
    args = parser.parse_args()
    if args.seeds < 1 or args.episodes < 1:
        parser.error("--seeds and --episodes must be positive integers.")

    cfg = {
        "episodes": args.episodes,
        "max_steps_per_episode": args.max_steps,
        "evaluation_episodes": args.eval_episodes,
        "alpha": 0.1,
        "gamma": 0.95,
        "epsilon_fixed": 0.2,
        "epsilon_min": 0.05,
        "epsilon_max": 0.8,
        "lambda_dispersion": 5.0,
        "curiosity_beta": 0.05,
        "planning_steps": args.planning_steps,
        "step_penalty": -0.01,
        "goal_reward": 1.0,
        "adaptive_epsilon_formula": "eps_min + (eps_max-eps_min)*exp(-lambda_dispersion*(max_a Q(s,a)-min_a Q(s,a)))",
        "curiosity_formula": "beta / sqrt(1 + N(s,a)); added to the learning reward for real and simulated updates",
        "notes": [
            "Initial deterministic single-maze baseline only; not the final multi-environment evaluation.",
            "Adaptive epsilon is a heuristic based on action-value dispersion, not a validated epistemic uncertainty estimate.",
            "Reported metrics use extrinsic reward; curiosity is used only in the learning target.",
            "Dyna model assumes deterministic transitions and stores only experienced state-action pairs.",
        ],
    }
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    maze_shortest = shortest_path_length()
    config_record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "config": cfg,
        "cli_args": vars(args),
        "algorithms": [
            "q_learning_fixed",
            "dyna_q_fixed",
            "dyna_q_curiosity",
            "dyna_q_adaptive",
            "dyna_q_curiosity_adaptive",
        ],
        "maze": MAZE,
        "start": START,
        "goal": GOAL,
        "shortest_path_length_moves": maze_shortest,
        "seed_values": list(range(args.seeds)),
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)

    all_episodes, all_evals = [], []
    start_time = time.time()
    for algorithm in config_record["algorithms"]:
        print(f"Running {algorithm} ...", flush=True)
        for seed in range(args.seeds):
            episodes, evals = run_one(seed, algorithm, cfg, output_dir)
            all_episodes.extend(episodes)
            all_evals.extend(evals)

    write_csv(output_dir / "per_episode_results.csv", all_episodes)
    write_csv(output_dir / "greedy_evaluation_results.csv", all_evals)

    summary_rows = []
    for algorithm in config_record["algorithms"]:
        rows = [r for r in all_episodes if r["algorithm"] == algorithm]
        evals = [r for r in all_evals if r["algorithm"] == algorithm]
        # Final 20% of training episodes; success rate is aggregated over all seed-episodes.
        tail_start = max(1, int(args.episodes * 0.8))
        tail = [r for r in rows if r["episode"] >= tail_start]
        eval_success = [r["success"] for r in evals]
        eval_steps_success = [r["steps"] for r in evals if r["success"] == 1]
        m_steps, sd_steps = mean_std([r["steps"] for r in tail])
        m_ret, sd_ret = mean_std([r["extrinsic_return"] for r in tail])
        m_eval_steps, sd_eval_steps = mean_std(eval_steps_success)
        summary_rows.append({
            "algorithm": algorithm,
            "training_tail_success_rate": statistics.mean([r["success"] for r in tail]) if tail else None,
            "training_tail_steps_mean": m_steps,
            "training_tail_steps_std": sd_steps,
            "training_tail_extrinsic_return_mean": m_ret,
            "training_tail_extrinsic_return_std": sd_ret,
            "greedy_eval_success_rate": statistics.mean(eval_success) if eval_success else None,
            "greedy_eval_successful_steps_mean": m_eval_steps,
            "greedy_eval_successful_steps_std": sd_eval_steps,
            "n_seeds": args.seeds,
            "n_eval_episodes_per_seed": args.eval_episodes,
            "shortest_path_length_moves": maze_shortest,
        })
    write_csv(output_dir / "summary_results.csv", summary_rows)

    # Plot mean training success and steps over episodes, aggregated over seeds.
    for metric, ylabel, filename in [
        ("success", "Success rate (fraction)", "learning_curve_success.png"),
        ("steps", "Steps per episode", "learning_curve_steps.png"),
        ("extrinsic_return", "Extrinsic return per episode", "learning_curve_return.png"),
    ]:
        plt.figure(figsize=(9, 5.5))
        for algorithm in config_record["algorithms"]:
            rows = [r for r in all_episodes if r["algorithm"] == algorithm]
            xs, means, lows, highs = [], [], [], []
            for ep in range(1, args.episodes + 1):
                vals = [r[metric] for r in rows if r["episode"] == ep]
                if vals:
                    xs.append(ep)
                    m, sd = mean_std(vals)
                    means.append(m)
                    lows.append(m - sd)
                    highs.append(m + sd)
            plt.plot(xs, means, label=algorithm)
            if len(xs) > 1:
                plt.fill_between(xs, lows, highs, alpha=0.12)
        plt.xlabel("Training episode")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} across independent seeds (shading = ±1 SD)")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=180)
        plt.close()

    with open(output_dir / "run_complete.json", "w", encoding="utf-8") as f:
        json.dump({
            "status": "completed",
            "elapsed_seconds": round(time.time() - start_time, 3),
            "output_files": [
                "config.json", "per_episode_results.csv", "greedy_evaluation_results.csv",
                "summary_results.csv", "learning_curve_success.png",
                "learning_curve_steps.png", "learning_curve_return.png",
                "artifacts/<algorithm>/seed_<seed>/q_table.npz",
                "artifacts/<algorithm>/seed_<seed>/learned_model.json",
                "artifacts/<algorithm>/seed_<seed>/run_metadata.json",
            ],
        }, f, indent=2)

    print("\nCompleted.")
    print(f"Shortest path in this maze: {maze_shortest} moves")
    print(f"Results saved to: {output_dir.resolve()}")
    print(f"Summary: {(output_dir / 'summary_results.csv').resolve()}")


if __name__ == "__main__":
    main()
