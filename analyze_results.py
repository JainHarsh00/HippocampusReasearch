#!/usr/bin/env python3
"""Seed-level analysis for run_experiments.py output folders. Standard library only."""
import argparse, csv, statistics
from pathlib import Path

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def mean_sd(vals):
    vals = [float(v) for v in vals if v not in ("", "None", "nan", "NaN")]
    if not vals:
        return "", ""
    return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0

def write_csv(path, rows):
    if not rows:
        print(f"No rows to write: {path}")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="Path to one timestamped run folder")
    args = p.parse_args()
    root = Path(args.run_dir)
    episodes = read_csv(root / "per_episode_results.csv")
    evaluations = read_csv(root / "greedy_evaluation_results.csv")
    config = read_csv(root / "summary_results.csv")  # used only to get algorithm names
    algorithms = [r["algorithm"] for r in config]
    n_episodes = max(int(r["episode"]) for r in episodes)
    tail_start = max(1, int(n_episodes * 0.8))
    seed_rows = []

    for algorithm in algorithms:
        seeds = sorted({int(r["seed"]) for r in episodes if r["algorithm"] == algorithm})
        for seed in seeds:
            train = [r for r in episodes if r["algorithm"] == algorithm and int(r["seed"]) == seed and int(r["episode"]) >= tail_start]
            ev = [r for r in evaluations if r["algorithm"] == algorithm and int(r["seed"]) == seed]
            successful_eval_steps = [int(r["steps"]) for r in ev if int(r["success"]) == 1]
            seed_rows.append({
                "algorithm": algorithm,
                "seed": seed,
                "training_tail_success_rate": statistics.mean(int(r["success"]) for r in train) if train else "",
                "training_tail_steps_mean": statistics.mean(int(r["steps"]) for r in train) if train else "",
                "training_tail_extrinsic_return_mean": statistics.mean(float(r["extrinsic_return"]) for r in train) if train else "",
                "greedy_eval_success_rate": statistics.mean(int(r["success"]) for r in ev) if ev else "",
                "greedy_eval_successful_steps_mean": statistics.mean(successful_eval_steps) if successful_eval_steps else "",
                "greedy_eval_successes": len(successful_eval_steps),
                "greedy_eval_episodes": len(ev),
            })
    write_csv(root / "per_seed_summary.csv", seed_rows)

    aggregate = []
    metric_fields = [
        "training_tail_success_rate",
        "training_tail_steps_mean",
        "training_tail_extrinsic_return_mean",
        "greedy_eval_success_rate",
        "greedy_eval_successful_steps_mean",
    ]
    for algorithm in algorithms:
        rows = [r for r in seed_rows if r["algorithm"] == algorithm]
        out = {"algorithm": algorithm, "n_seeds": len(rows)}
        for field in metric_fields:
            m, sd = mean_sd([r[field] for r in rows])
            out[field + "_mean_across_seeds"] = m
            out[field + "_sd_across_seeds"] = sd
        aggregate.append(out)
    write_csv(root / "seed_level_summary.csv", aggregate)
    print("Wrote:")
    print(" ", root / "per_seed_summary.csv")
    print(" ", root / "seed_level_summary.csv")
    print("Interpretation: SDs here are across independent seeds. These are descriptive summaries; use a pre-specified inferential test if making significance claims.")

if __name__ == "__main__":
    main()
