#!/usr/bin/env python3
"""
Controlled diagnostic sweep for the adaptive-epsilon mechanism.

Imports run_experiments.py from this same folder. Compares Dyna-Q fixed epsilon
against Dyna-Q adaptive epsilon over a predeclared lambda grid, using matched seeds
and otherwise identical settings. No parameters are selected automatically as "best";
inspect the saved results before choosing a follow-up validation experiment.
"""
import argparse, csv, json, statistics, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import run_experiments as base

def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def mean_sd(values):
    vals=[float(v) for v in values if v is not None and v != ""]
    if not vals: return "", ""
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals)>1 else 0.0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",type=int,default=10)
    ap.add_argument("--episodes",type=int,default=200)
    ap.add_argument("--max-steps",type=int,default=150)
    ap.add_argument("--planning-steps",type=int,default=5)
    ap.add_argument("--eval-episodes",type=int,default=10)
    ap.add_argument("--output",default="results")
    ap.add_argument("--lambdas",default="0,0.5,1,2,5,10",
                    help="Comma-separated lambda grid; default 0,0.5,1,2,5,10")
    args=ap.parse_args()
    if args.seeds<2: ap.error("Use at least 2 seeds; 10 or more is recommended.")
    if args.episodes<1 or args.max_steps<1 or args.eval_episodes<1: ap.error("Episode and step counts must be positive.")
    lambdas=[float(x.strip()) for x in args.lambdas.split(",") if x.strip()]
    if not lambdas or any(x<0 for x in lambdas): ap.error("Lambdas must be a nonempty list of nonnegative values.")

    cfg={
        "episodes":args.episodes,
        "max_steps_per_episode":args.max_steps,
        "evaluation_episodes":args.eval_episodes,
        "alpha":0.1,"gamma":0.95,
        "epsilon_fixed":0.2,
        "epsilon_min":0.05,"epsilon_max":0.8,
        "lambda_dispersion":5.0,
        "curiosity_beta":0.05,
        "planning_steps":args.planning_steps,
        "step_penalty":-0.01,"goal_reward":1.0,
        "adaptive_epsilon_formula":"eps_min + (eps_max-eps_min)*exp(-lambda_dispersion*(max_a Q(s,a)-min_a Q(s,a)))",
        "curiosity_formula":"beta / sqrt(1 + N(s,a)); not used in this diagnostic",
        "diagnostic_note":"Lambda sweep isolates adaptive epsilon without curiosity. This is a diagnostic, not confirmatory evidence.",
    }
    run_id=datetime.now().strftime("diagnostic_%Y%m%d_%H%M%S")
    out=Path(args.output)/run_id
    out.mkdir(parents=True,exist_ok=False)
    algorithms=["dyna_q_fixed"]+[f"dyna_q_adaptive_lambda_{lam:g}" for lam in lambdas]
    config={
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "run_id":run_id,"cli_args":vars(args),"base_config":cfg,
        "algorithms":algorithms,"lambda_grid":lambdas,
        "matched_seed_values":list(range(args.seeds)),
        "maze":base.MAZE,"start":base.START,"goal":base.GOAL,
        "shortest_path_length_moves":base.shortest_path_length(),
        "design":"Fixed Dyna-Q epsilon=0.2 vs adaptive epsilon, lambda sweep; all other settings held constant.",
        "adaptive_epsilon":"eps_min + (eps_max-eps_min)*exp(-lambda*Q_dispersion)",
        "interpretation_warning":"Q dispersion measures action-value separation, not calibrated epistemic uncertainty.",
    }
    with open(out/"diagnostic_config.json","w",encoding="utf-8") as f: json.dump(config,f,indent=2)
    all_ep=[]; all_ev=[]; seed_summaries=[]
    t0=time.time()
    for alg in algorithms:
        for seed in range(args.seeds):
            run_cfg=cfg.copy()
            if alg=="dyna_q_fixed":
                actual_alg="dyna_q_fixed"
            else:
                actual_alg="dyna_q_adaptive"
                lam=float(alg.rsplit("_",1)[1])
                run_cfg["lambda_dispersion"]=lam
            print(f"{alg}, seed {seed}",flush=True)
            eps, ev=base.run_one(seed,actual_alg,run_cfg,out / "artifacts_by_condition" / alg)
            # Re-label so lambda conditions remain distinct in all tables.
            for row in eps:
                row["condition"]=alg
                row["lambda_dispersion"]="" if alg=="dyna_q_fixed" else float(alg.rsplit("_",1)[1])
                row["algorithm"]=alg
            for row in ev:
                row["condition"]=alg
                row["lambda_dispersion"]="" if alg=="dyna_q_fixed" else float(alg.rsplit("_",1)[1])
                row["algorithm"]=alg
            all_ep.extend(eps); all_ev.extend(ev)
            tail_start=max(1,int(args.episodes*0.8))
            tail=[r for r in eps if int(r["episode"])>=tail_start]
            successful=[int(r["steps"]) for r in ev if int(r["success"])==1]
            seed_summaries.append({
                "condition":alg,"seed":seed,
                "lambda_dispersion":"" if alg=="dyna_q_fixed" else float(alg.rsplit("_",1)[1]),
                "training_tail_success_rate":statistics.mean(int(r["success"]) for r in tail),
                "training_tail_steps_mean":statistics.mean(float(r["steps"]) for r in tail),
                "training_tail_extrinsic_return_mean":statistics.mean(float(r["extrinsic_return"]) for r in tail),
                "training_tail_mean_epsilon":statistics.mean(float(r["mean_epsilon"]) for r in tail if r["mean_epsilon"] not in ("",None)),
                "greedy_eval_success_rate":statistics.mean(int(r["success"]) for r in ev),
                "greedy_eval_successful_steps_mean":statistics.mean(successful) if successful else "",
                "greedy_eval_success_count":len(successful),
                "greedy_eval_episode_count":len(ev),
            })
    write_csv(out/"per_episode_results.csv",all_ep)
    write_csv(out/"greedy_evaluation_results.csv",all_ev)
    write_csv(out/"per_seed_summary.csv",seed_summaries)
    metrics=["training_tail_success_rate","training_tail_steps_mean","training_tail_extrinsic_return_mean",
             "training_tail_mean_epsilon","greedy_eval_success_rate","greedy_eval_successful_steps_mean"]
    aggregate=[]
    for alg in algorithms:
        rows=[r for r in seed_summaries if r["condition"]==alg]
        rec={"condition":alg,"lambda_dispersion":rows[0]["lambda_dispersion"],"n_seeds":len(rows)}
        for m in metrics:
            mean,sd=mean_sd([r[m] for r in rows])
            rec[m+"_mean_across_seeds"]=mean
            rec[m+"_sd_across_seeds"]=sd
        aggregate.append(rec)
    write_csv(out/"seed_level_summary.csv",aggregate)

    # Curves aggregate by episode across matched seeds; shade ±1 SD across seeds.
    for metric,ylabel,filename in [
        ("success","Success rate","lambda_diagnostic_success.png"),
        ("steps","Steps per episode","lambda_diagnostic_steps.png"),
        ("mean_epsilon","Mean epsilon","lambda_diagnostic_epsilon.png"),
    ]:
        plt.figure(figsize=(10,6))
        for alg in algorithms:
            rows=[r for r in all_ep if r["algorithm"]==alg]
            xs=[]; ms=[]; lows=[]; highs=[]
            for ep in range(1,args.episodes+1):
                # First average within each seed is one episode observation; one row per seed/episode.
                vals=[float(r[metric]) for r in rows if int(r["episode"])==ep and r[metric] not in ("",None)]
                if vals:
                    m,sd=mean_sd(vals); xs.append(ep); ms.append(m); lows.append(m-sd); highs.append(m+sd)
            plt.plot(xs,ms,label=alg)
            if len(xs)>1: plt.fill_between(xs,lows,highs,alpha=0.12)
        plt.xlabel("Training episode"); plt.ylabel(ylabel)
        plt.title(f"{ylabel} by condition (shading = ±1 SD across seeds)")
        plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(out/filename,dpi=180); plt.close()
    with open(out/"run_complete.json","w",encoding="utf-8") as f:
        json.dump({"status":"completed","elapsed_seconds":round(time.time()-t0,2),
          "files":["diagnostic_config.json","per_episode_results.csv","greedy_evaluation_results.csv",
          "per_seed_summary.csv","seed_level_summary.csv","lambda_diagnostic_success.png",
          "lambda_diagnostic_steps.png","lambda_diagnostic_epsilon.png",
          "artifacts/<algorithm>/seed_<seed>/q_table.npz",
          "artifacts/<algorithm>/seed_<seed>/learned_model.json"]},f,indent=2)
    print("\nDiagnostic complete:",out.resolve())
    print("Shortest path:",base.shortest_path_length(),"moves")
    print("Send back the entire timestamped diagnostic folder as a ZIP.")

if __name__=="__main__": main()
