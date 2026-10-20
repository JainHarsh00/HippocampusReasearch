#!/usr/bin/env python3
"""Leakage-audited reliability-aware Dyna-Q ablation experiment.

Reliability is computed only from predictions made by the agent's learned empirical
transition model BEFORE the real outcome is observed, compared with that subsequent
outcome. The reliability estimator never receives the corruption mask or true
transition function. Controlled corruption modifies the planning prediction only.
"""
from __future__ import annotations
import argparse, csv, json, math, random, statistics, time
from datetime import datetime, timezone
from pathlib import Path
from collections import deque, Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACTIONS = {0:(-1,0), 1:(0,1), 2:(1,0), 3:(0,-1)}
ACTION_NAMES = ["up","right","down","left"]
MAZES = {
 "maze_a": ["S...#...", ".##.#.#.", "...#....", ".#...##.", ".###.#..", "...#.#..", "##...#..", ".......G"],
 "maze_b": ["S.#.....", "...#.#..", ".#...#..", ".###....", "...##.#.", "#.....#.", "..#.....", ".....#.G"],
 "maze_c": ["S....#..", "##.#....", "...#.#..", ".#...#..", ".#.#....", "...##.#.", ".#......", "...#...G"],
}

def parse_maze(rows):
    h, w = len(rows), len(rows[0])
    if any(len(row) != w for row in rows):
        raise ValueError("Maze rows must have equal width")
    starts = [(r,c) for r,row in enumerate(rows) for c,ch in enumerate(row) if ch == "S"]
    goals = [(r,c) for r,row in enumerate(rows) for c,ch in enumerate(row) if ch == "G"]
    if len(starts) != 1 or len(goals) != 1:
        raise ValueError("Each maze must have exactly one S and one G")
    start, goal = starts[0], goals[0]
    walls = {(r,c) for r,row in enumerate(rows) for c,ch in enumerate(row) if ch == "#"}
    valid = [(r,c) for r in range(h) for c in range(w) if (r,c) not in walls]
    def true_next(s,a):
        dr,dc = ACTIONS[int(a)]
        n = (s[0]+dr,s[1]+dc)
        return n if 0 <= n[0] < h and 0 <= n[1] < w and n not in walls else s
    q = deque([(start,0)]); seen = {start}; dist = None
    while q:
        s,d = q.popleft()
        if s == goal:
            dist = d; break
        for a in ACTIONS:
            n = true_next(s,a)
            if n not in seen:
                seen.add(n); q.append((n,d+1))
    if dist is None:
        raise ValueError("Unsolvable maze")
    return {"rows":rows,"h":h,"w":w,"start":start,"goal":goal,"walls":walls,
            "valid":valid,"true_next":true_next,"shortest":dist}
WORLD = {name:parse_maze(rows) for name,rows in MAZES.items()}

def sid(s,w): return s[0]*w+s[1]
def state(i,w): return (i//w,i%w)

def eps_for(qrow,cfg,adaptive):
    if not adaptive: return cfg["epsilon_fixed"]
    dispersion = float(np.max(qrow)-np.min(qrow))
    return float(np.clip(cfg["epsilon_min"]+(cfg["epsilon_max"]-cfg["epsilon_min"])*
                         math.exp(-cfg["lambda_dispersion"]*dispersion),
                         cfg["epsilon_min"],cfg["epsilon_max"]))

def choose(q,s,rng,cfg,adaptive,greedy=False):
    eps = 0.0 if greedy else eps_for(q[s],cfg,adaptive)
    if rng.random() < eps: return rng.randrange(4),eps
    best = np.flatnonzero(np.isclose(q[s],np.max(q[s])))
    return int(rng.choice(best.tolist())),eps

def q_update(q,s,a,r,ns,done,cfg,weight=1.0):
    target = r if done else r+cfg["gamma"]*float(np.max(q[ns]))
    q[s,a] += cfg["alpha"]*float(np.clip(weight,0.0,1.0))*(target-q[s,a])

class GridWorld:
    def __init__(self,world,rng):
        self.world=world; self.rng=rng; self.s=world["start"]
    def reset(self):
        self.s=self.world["start"]; return self.s
    def step(self,a):
        ns=self.world["true_next"](self.s,a)
        done=(ns==self.world["goal"])
        self.s=ns
        return ns,(1.0 if done else -0.01),done

def build_bias_order(world,seed):
    """Predefine nested corruption masks; labels are only used by model intervention."""
    candidates=[]
    for s in world["valid"]:
        if s==world["goal"]: continue
        for a in ACTIONS:
            true_ns=world["true_next"](s,a)
            if true_ns==world["goal"]: continue
            candidates.append((sid(s,world["w"]),a,sid(true_ns,world["w"])))
    rng=random.Random(seed*1000003 + sum(ord(c) for c in "".join(world["rows"]))*97 + 8081)
    rng.shuffle(candidates)
    non_goal=[sid(s,world["w"]) for s in world["valid"] if s!=world["goal"]]
    mapping={}
    for si,a,true_ni in candidates:
        opts=[x for x in non_goal if x!=true_ni]
        mapping[(si,a)]=rng.choice(opts)
    return candidates,mapping

def model_prediction(pair, model, w, corruption_enabled):
    """Predict from empirical learned outcomes only. Never queries world true_next."""
    outcomes=model.get(pair,[])
    if not outcomes:
        return None
    counts=Counter(int(item[0]) for item in outcomes)
    # Stable deterministic tie-break; reward/done come from empirical outcomes too.
    next_id=sorted(counts, key=lambda x:(-counts[x],x))[0]
    relevant=[item for item in outcomes if int(item[0])==next_id]
    done_counts=Counter(bool(item[2]) for item in relevant)
    done=sorted(done_counts,key=lambda x:(-done_counts[x],x))[0]
    reward=float(np.mean([float(item[1]) for item in relevant]))
    return next_id,reward,done

def planning_prediction(pair, model, corruption_mask, corrupt_map, w):
    """Construct the prediction consumed by planning. Reliability code sees only result."""
    pred=model_prediction(pair,model,w,False)
    if pred is None: return None
    ni,r,done=pred
    if pair in corruption_mask and not done:
        ni=corrupt_map[pair]
    return int(ni),float(r),bool(done)

def run_one(world_name,world,seed,bias_fraction,algorithm,cfg,out):
    w=world["w"]; n_states=world["h"]*w
    order,corrupt_map=build_bias_order(world,seed)
    n_corrupt=round(len(order)*bias_fraction)
    corruption_mask={(si,a) for si,a,_ in order[:n_corrupt]}
    # Independent RNG streams; algorithm-specific action randomness, shared environment seed.
    agent_rng=random.Random(seed+100003+sum(ord(c) for c in algorithm))
    env_rng=random.Random(seed+200003+sum(ord(c) for c in world_name))
    eval_env_rng=random.Random(seed+300007+sum(ord(c) for c in world_name))
    eval_action_rng=random.Random(seed+400009+sum(ord(c) for c in world_name))
    env=GridWorld(world,env_rng); eval_env=GridWorld(world,eval_env_rng)
    q=np.zeros((n_states,4)); model={}; experienced=[]; seen=set(); visits={}
    reliability_counts={}  # pair -> [correct predictions, evaluated predictions]
    episode_rows=[]; planning_reliabilities=[]; prediction_audit=[]
    planning=algorithm!="q_learning"
    adaptive=algorithm=="dyna_q_adaptive"
    uses_reliability=algorithm in ("dyna_q_reliability","dyna_q_no_weight","dyna_q_shuffled_reliability")
    for ep in range(1,cfg["episodes"]+1):
        s=env.reset(); steps=0; ret=0.0; success=0; epss=[]; plan_updates=0
        while steps<cfg["max_steps_per_episode"]:
            si=sid(s,w); a,eps=choose(q,si,agent_rng,cfg,adaptive); epss.append(eps)
            pair=(si,a)
            # Crucial order: generate prediction using only past model data, then step env.
            pred_before=planning_prediction(pair,model,corruption_mask,corrupt_map,w) if model.get(pair) else None
            ns,r,done=env.step(a); ni=sid(ns,w)
            q_update(q,si,a,r,ni,done,cfg)  # real update uses real observation
            visits[pair]=visits.get(pair,0)+1
            if pair not in seen:
                seen.add(pair); experienced.append(pair)
            if uses_reliability and pred_before is not None:
                pred_ni=pred_before[0]
                correct=(int(pred_ni)==int(ni))
                m,n=reliability_counts.get(pair,(0,0))
                reliability_counts[pair]=(m+int(correct),n+1)
                prediction_audit.append({"state_id":si,"action":a,"predicted_next_state_id":pred_ni,
                                         "observed_next_state_id":ni,"prediction_correct":int(correct),
                                         "prediction_made_before_observation":True})
            model.setdefault(pair,[]).append((ni,r,done))
            if planning and experienced:
                for _ in range(cfg["planning_steps"]):
                    ps,pa=agent_rng.choice(experienced)
                    # The planning prediction is derived from learned model outcomes;
                    # corruption is applied as a separate intervention.
                    pp=planning_prediction((ps,pa),model,corruption_mask,corrupt_map,w)
                    if pp is None:
                        continue
                    pni,pr,pdone=pp
                    weight=1.0
                    if uses_reliability:
                        m,n=reliability_counts.get((ps,pa),(0,0))
                        # No evidence -> zero trust; otherwise Laplace-smoothed past agreement.
                        rel=(m+1.0)/(n+2.0) if n>0 else 0.0
                        if algorithm=="dyna_q_shuffled_reliability":
                            pool=[(m2+1.0)/(n2+2.0) for m2,n2 in reliability_counts.values() if n2>0]
                            if pool:
                                shuf_rng=random.Random(seed+700001+ep*101+ps*7+pa)
                                rel=pool[shuf_rng.randrange(len(pool))]
                        planning_reliabilities.append(rel)
                        if algorithm!="dyna_q_no_weight": weight=rel
                    q_update(q,ps,pa,pr,pni,pdone,cfg,weight=weight)
                    plan_updates+=1
            ret+=r; steps+=1; s=ns
            if done: success=1; break
        episode_rows.append({"maze":world_name,"bias_fraction":bias_fraction,"algorithm":algorithm,
          "seed":seed,"episode":ep,"steps":steps,"success":success,"extrinsic_return":ret,
          "mean_epsilon":float(np.mean(epss)) if epss else None,
          "planning_updates_applied":plan_updates,
          "mean_planning_reliability":float(np.mean(planning_reliabilities)) if planning_reliabilities else None,
          "observed_pairs":len(seen),"corrupted_pairs_predefined":len(corruption_mask),
          "corrupted_pairs_observed":len(corruption_mask & seen)})
    eval_rows=[]
    for e in range(1,cfg["evaluation_episodes"]+1):
        s=eval_env.reset(); steps=0; ret=0.0; success=0
        while steps<cfg["max_steps_per_episode"]:
            si=sid(s,w); a,_=choose(q,si,eval_action_rng,cfg,False,greedy=True)
            ns,r,done=eval_env.step(a); ret+=r; steps+=1; s=ns
            if done: success=1; break
        eval_rows.append({"maze":world_name,"bias_fraction":bias_fraction,"algorithm":algorithm,
                          "seed":seed,"evaluation_episode":e,"steps":steps,"success":success,
                          "extrinsic_return":ret})
    art=out/"artifacts"/world_name/f"bias_{bias_fraction:.2f}"/algorithm/f"seed_{seed:04d}"
    art.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(art/"q_table.npz",q_table=q,action_names=np.array(ACTION_NAMES))
    model_rows=[]
    for (si,a),outs in sorted(model.items()):
        for oi,(ni,r,d) in enumerate(outs):
            pp=planning_prediction((si,a),model,corruption_mask,corrupt_map,w)
            pred_ni=pp[0] if pp is not None else None
            model_rows.append({"state_id":si,"state":list(state(si,w)),"action":ACTION_NAMES[a],
              "outcome_index":oi,"observed_next_state":list(state(ni,w)),
              "planning_next_state":list(state(pred_ni,w)) if pred_ni is not None else None,
              "reward":r,"done":d,"visit_count":visits[(si,a)],
              "corrupted_prediction":bool((si,a) in corruption_mask and not d)})
    (art/"learned_model.json").write_text(json.dumps(model_rows,indent=2),encoding="utf-8")
    (art/"prediction_audit.json").write_text(json.dumps(prediction_audit,indent=2),encoding="utf-8")
    meta={"maze":world_name,"seed":seed,"bias_fraction":bias_fraction,"algorithm":algorithm,
      "config":cfg,"shortest_path_length_moves":world["shortest"],
      "reliability_counts":{f"{si}:{a}":{"matches":m,"observations":n,"smoothed_reliability":(m+1)/(n+2)}
                            for (si,a),(m,n) in reliability_counts.items()} if uses_reliability else None,
      "prediction_audit_count":len(prediction_audit),
      "prediction_audit_all_pre_observation":all(x["prediction_made_before_observation"] for x in prediction_audit),
      "predefined_corrupted_pairs":len(corruption_mask),"observed_corrupted_pairs":len(corruption_mask & seen),
      "rng_design":"separate agent, training environment, evaluation environment, and greedy action streams",
      "reliability_information_boundary":"learned empirical transition outcomes and prior prediction-observation pairs only; no true transition function or corruption-mask labels are passed to reliability estimator"}
    (art/"run_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    return episode_rows,eval_rows

def write_csv(path,rows):
    if rows:
        with path.open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def mean_sd(vals):
    vals=[float(v) for v in vals if v is not None]
    return (statistics.mean(vals),statistics.stdev(vals) if len(vals)>1 else 0.0) if vals else (None,None)

def main():
    parser=argparse.ArgumentParser(description="Leakage-audited reliability-aware Dyna-Q ablation")
    parser.add_argument("--seeds",type=int,default=10); parser.add_argument("--episodes",type=int,default=200)
    parser.add_argument("--max-steps",type=int,default=150); parser.add_argument("--planning-steps",type=int,default=5)
    parser.add_argument("--eval-episodes",type=int,default=10); parser.add_argument("--output",default="results")
    parser.add_argument("--bias-levels",default="0,0.05,0.10,0.20,0.30")
    args=parser.parse_args()
    if min(args.seeds,args.episodes,args.max_steps,args.eval_episodes)<1 or args.planning_steps<0:
        parser.error("counts must be positive (planning-steps may be zero)")
    levels=[float(x.strip()) for x in args.bias_levels.split(",")]
    if not levels or any(x<0 or x>0.8 for x in levels) or len(set(levels))!=len(levels):
        parser.error("bias levels must be unique and between 0 and 0.8")
    levels=sorted(levels)
    cfg={"episodes":args.episodes,"max_steps_per_episode":args.max_steps,"evaluation_episodes":args.eval_episodes,
      "alpha":0.1,"gamma":0.95,"epsilon_fixed":0.2,"epsilon_min":0.05,"epsilon_max":0.8,
      "lambda_dispersion":5.0,"planning_steps":args.planning_steps,"step_penalty":-0.01,"goal_reward":1.0,
      "adaptive_epsilon_formula":"eps_min + (eps_max-eps_min)*exp(-lambda_dispersion*(max_a Q(s,a)-min_a Q(s,a)))",
      "model_bias_scope":"planning predictions only; real updates use observed transitions",
      "reliability_method":"empirical modal prediction from learned outcomes, scored against subsequent observation; reliability weights later planning updates; no true transition function or corruption labels available to estimator",
      "mask_design":"nested corruption masks predefined per maze and seed; labels used only by planning-model intervention",
      "rng_design":"separate agent, training environment, evaluation environment, and greedy action streams"}
    algs=["q_learning","dyna_q_fixed","dyna_q_adaptive","dyna_q_reliability","dyna_q_no_weight","dyna_q_shuffled_reliability"]
    run_id=datetime.now().strftime("reliability_audit_%Y%m%d_%H%M%S")
    out=Path(args.output)/run_id; out.mkdir(parents=True,exist_ok=False)
    record={"created_utc":datetime.now(timezone.utc).isoformat(),"run_id":run_id,"cli_args":vars(args),
      "config":cfg,"bias_levels":levels,"algorithms":algs,
      "mazes":{k:{"rows":v["rows"],"start":v["start"],"goal":v["goal"],"shortest_path_length_moves":v["shortest"]} for k,v in WORLD.items()},
      "seed_values":list(range(args.seeds)),"planned_runs":len(WORLD)*len(levels)*len(algs)*args.seeds,
      "interpretation_guardrails":["Corruption labels are hidden from reliability estimator.","Reliability predictions are generated from learned outcomes before current outcome is observed.","Unseen pairs have no reliability evidence; reliability weight defaults to zero.","Three fixed mazes do not establish generality beyond these environments."]}
    (out/"config.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    episode_rows=[]; eval_rows=[]; t0=time.time(); total=record["planned_runs"]; completed=0
    for maze_name,world in WORLD.items():
      for level in levels:
        for alg in algs:
          for seed in range(args.seeds):
            print(f"[{completed+1}/{total}] maze={maze_name} bias={level:.2f} algorithm={alg} seed={seed}",flush=True)
            er,vr=run_one(maze_name,world,seed,level,alg,cfg,out)
            episode_rows.extend(er); eval_rows.extend(vr); completed+=1
    write_csv(out/"per_episode_results.csv",episode_rows); write_csv(out/"greedy_evaluation_results.csv",eval_rows)
    tail_start=max(1,int(args.episodes*0.8)); seed_rows=[]
    for maze_name in WORLD:
      for level in levels:
        for alg in algs:
          for seed in range(args.seeds):
            rr=[x for x in episode_rows if x["maze"]==maze_name and x["bias_fraction"]==level and x["algorithm"]==alg and x["seed"]==seed and x["episode"]>=tail_start]
            vv=[x for x in eval_rows if x["maze"]==maze_name and x["bias_fraction"]==level and x["algorithm"]==alg and x["seed"]==seed]
            seed_rows.append({"maze":maze_name,"bias_fraction":level,"algorithm":alg,"seed":seed,
              "tail_steps_mean":statistics.mean(x["steps"] for x in rr),
              "tail_success_rate":statistics.mean(x["success"] for x in rr),
              "tail_extrinsic_return_mean":statistics.mean(x["extrinsic_return"] for x in rr),
              "eval_success_rate":statistics.mean(x["success"] for x in vv),
              "eval_successful_steps_mean":statistics.mean(x["steps"] for x in vv if x["success"]) if any(x["success"] for x in vv) else None})
    write_csv(out/"seed_level_summary.csv",seed_rows)
    summary=[]
    for maze_name in WORLD:
      for level in levels:
        for alg in algs:
          rr=[x for x in seed_rows if x["maze"]==maze_name and x["bias_fraction"]==level and x["algorithm"]==alg]
          row={"maze":maze_name,"bias_fraction":level,"algorithm":alg,"n_seeds":args.seeds}
          for metric in ["tail_steps_mean","tail_success_rate","tail_extrinsic_return_mean","eval_success_rate","eval_successful_steps_mean"]:
            m,sd=mean_sd([x[metric] for x in rr]); row[metric+"_mean"]=m; row[metric+"_sd_across_seeds"]=sd
          summary.append(row)
    write_csv(out/"summary_by_maze.csv",summary)
    pooled=[]
    for level in levels:
      for alg in algs:
        rr=[x for x in seed_rows if x["bias_fraction"]==level and x["algorithm"]==alg]
        row={"bias_fraction":level,"algorithm":alg,"n_maze_seed_units":len(rr)}
        for metric in ["tail_steps_mean","tail_success_rate","tail_extrinsic_return_mean","eval_success_rate"]:
          m,sd=mean_sd([x[metric] for x in rr]); row[metric+"_mean"]=m; row[metric+"_sd_across_maze_seed_units"]=sd
        pooled.append(row)
    write_csv(out/"pooled_summary.csv",pooled)
    for metric,ylabel,fn in [("eval_success_rate","Greedy evaluation success","reliability_eval_success.png"),
                             ("tail_steps_mean","Final-window training steps","reliability_tail_steps.png"),
                             ("tail_success_rate","Final-window training success","reliability_train_success.png")]:
      fig,ax=plt.subplots(figsize=(8.5,5.5))
      for alg in algs:
        means=[]; sds=[]
        for level in levels:
          vals=[x[metric] for x in seed_rows if x["bias_fraction"]==level and x["algorithm"]==alg]
          m,sd=mean_sd(vals); means.append(m); sds.append(sd)
        ax.errorbar([100*x for x in levels],means,yerr=sds,marker="o",capsize=3,label=alg)
      ax.set_xlabel("Predefined model-bias level (%)"); ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend()
      fig.tight_layout(); fig.savefig(out/fn,dpi=180); plt.close(fig)
    (out/"run_complete.json").write_text(json.dumps({"status":"completed","elapsed_seconds":round(time.time()-t0,2),
      "completed_runs":completed,"planned_runs":total,"files":["config.json","per_episode_results.csv",
      "greedy_evaluation_results.csv","seed_level_summary.csv","summary_by_maze.csv","pooled_summary.csv",
      "reliability_eval_success.png","reliability_tail_steps.png","reliability_train_success.png",
      "artifacts/<maze>/bias_<level>/<algorithm>/seed_<seed>/{q_table.npz,learned_model.json,prediction_audit.json,run_metadata.json}"]},indent=2),encoding="utf-8")
    print("\nCompleted."); print("Output:",out.resolve()); print("Pooled summary:",(out/"pooled_summary.csv").resolve())

if __name__=="__main__":
    main()
