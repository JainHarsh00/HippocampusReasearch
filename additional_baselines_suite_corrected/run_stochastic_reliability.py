#!/usr/bin/env python3
"""Probability-aware reliability-weighted Dyna-Q experiment under stochastic movement.

Controlled corruption is applied only to planning predictions for selected state-action
pairs. The estimator receives prior predictive distributions and subsequent outcomes, but
never the corruption mask or the environment transition function. Real Q-updates always
use observed transitions at weight 1.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, platform, random, statistics, time, heapq
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter, deque
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ACTIONS = {0:(-1,0), 1:(0,1), 2:(1,0), 3:(0,-1)}
ACTION_NAMES = ["up", "right", "down", "left"]
MAZES = {
 "large_12x12_short": [
  "S..#..#..#..", "..##........", "....#...#.#.", "..#..#....#.",
  "............", "..#...#.....", "............", "............",
  ".#........#.", "#...###.#...", ".#..........", "..#.......#G"],
 "large_16x16_long": [
  "S..#.....#.#.#.#", "...#..#.#.......", "..#........#.##.", ".......##...#...",
  "...#..#.#.##..##", "..#.#.###.##....", "#.....#####..#..", "#...#####.#.#...",
  ".....#.###.#....", "#...#...#.......", "..#..#.#.##...##", "##..#.....#.....",
  ".#...#.....#..##", "#.######....#..#", "...##....#...#..", "...#.#...#.....G"]
}

def parse_maze(rows):
    h, w = len(rows), len(rows[0])
    if any(len(row) != w for row in rows):
        raise ValueError("Maze rows must have equal width")
    starts = [(r,c) for r,row in enumerate(rows) for c,ch in enumerate(row) if ch == "S"]
    goals = [(r,c) for r,row in enumerate(rows) for c,ch in enumerate(row) if ch == "G"]
    if len(starts) != 1 or len(goals) != 1:
        raise ValueError("Each maze must contain exactly one S and one G")
    start, goal = starts[0], goals[0]
    walls = {(r,c) for r,row in enumerate(rows) for c,ch in enumerate(row) if ch == "#"}
    valid = [(r,c) for r in range(h) for c in range(w) if (r,c) not in walls]
    def true_next(s,a):
        dr,dc = ACTIONS[int(a)]
        n=(s[0]+dr,s[1]+dc)
        return n if 0 <= n[0] < h and 0 <= n[1] < w and n not in walls else s
    q=deque([(start,0)]); seen={start}; dist=None
    while q:
        s,d=q.popleft()
        if s==goal:
            dist=d; break
        for a in ACTIONS:
            ns=true_next(s,a)
            if ns not in seen:
                seen.add(ns); q.append((ns,d+1))
    if dist is None:
        raise ValueError("Unsolvable maze")
    return {"rows":rows,"h":h,"w":w,"start":start,"goal":goal,"walls":walls,
            "valid":valid,"true_next":true_next,"shortest":dist}
WORLD={name:parse_maze(rows) for name,rows in MAZES.items()}

def sid(s,w): return s[0]*w+s[1]
def state(i,w): return (i//w,i%w)

def eps_for(qrow,cfg,adaptive):
    if not adaptive: return cfg["epsilon_fixed"]
    dq=float(np.max(qrow)-np.min(qrow))
    return float(np.clip(cfg["epsilon_min"]+(cfg["epsilon_max"]-cfg["epsilon_min"])*
                         math.exp(-cfg["lambda_dispersion"]*dq),
                         cfg["epsilon_min"],cfg["epsilon_max"]))

def choose(q,s,rng,cfg,adaptive,greedy=False):
    eps=0.0 if greedy else eps_for(q[s],cfg,adaptive)
    if rng.random() < eps: return rng.randrange(4),eps
    best=np.flatnonzero(np.isclose(q[s],np.max(q[s])))
    return int(rng.choice(best.tolist())),eps

def q_update(q,s,a,r,ns,done,cfg,weight=1.0):
    target=r if done else r+cfg["gamma"]*float(np.max(q[ns]))
    q[s,a] += cfg["alpha"]*float(np.clip(weight,0.0,1.0))*(target-q[s,a])

class GridWorld:
    def __init__(self,world,rng,slip):
        self.world=world; self.rng=rng; self.slip=slip; self.s=world["start"]
    def reset(self): self.s=self.world["start"]; return self.s
    def step(self,a):
        u=self.rng.random()
        if u < 1.0-self.slip:
            executed=a
        elif u < 1.0-self.slip/2.0:
            executed=(a-1)%4
        else:
            executed=(a+1)%4
        ns=self.world["true_next"](self.s,executed)
        done=(ns==self.world["goal"])
        self.s=ns
        return ns,(1.0 if done else -0.01),done

def build_corruption_design(world,seed):
    """Pair masks are fixed per map/seed; wrong destinations are outside local slip support."""
    candidates=[]; wrong_map={}
    valid_ids=[sid(s,world["w"]) for s in world["valid"]]
    goal_id=sid(world["goal"],world["w"])
    text="".join(world["rows"])
    rng=random.Random(seed*1000003 + sum(ord(c) for c in text)*97 + 93017)
    for s in world["valid"]:
        if s==world["goal"]: continue
        si=sid(s,world["w"])
        for a in ACTIONS:
            # Stochastic execution can choose intended or either perpendicular direction.
            local={sid(world["true_next"](s,ea),world["w"]) for ea in (a,(a-1)%4,(a+1)%4)}
            # Never corrupt a pair whose possible slip outcome can terminate the episode.
            if goal_id in local: continue
            candidates.append((si,a))
            alternatives=[x for x in valid_ids if x != goal_id and x not in local]
            if not alternatives: raise ValueError(f"No valid distractor for state-action pair {(si,a)}")
            wrong_map[(si,a)]=rng.choice(alternatives)
    rng.shuffle(candidates)
    # Keep a deterministic, nested pair order and an associated distractor for each pair.
    return candidates,wrong_map

def empirical_distribution(pair,model,corruption_mask,wrong_map):
    counts=model.get(pair)
    if not counts: return None
    if pair in corruption_mask: return {wrong_map[pair]:1.0}
    total=sum(counts.values())
    return {int(ns):n/total for ns,n in counts.items()}

def sample_distribution(dist,rng):
    keys=sorted(dist)
    vals=[dist[k] for k in keys]
    return int(rng.choices(keys,weights=vals,k=1)[0])

def nll_from_forecast(dist,observed_ns,n_valid):
    """Capped prequential log loss; uniform-over-valid-cells loss is the normalization baseline."""
    p=float(dist.get(int(observed_ns),0.0))
    cap=math.log(max(2,n_valid))
    return min(-math.log(max(p,1e-12)),cap)

def reliability_weight(stats,n_valid):
    n=stats.get("n",0)
    if n<=0: return 0.0
    mean_loss=stats["loss_sum"]/n
    # Predictive skill relative to a uniform next-state forecast, shrunk toward zero
    # for few scored predictions. This is an empirical weight, not calibrated probability.
    skill=float(np.clip(1.0-mean_loss/math.log(max(2,n_valid)),0.0,1.0))
    return float(np.clip((n/(n+2.0))*skill,0.0,1.0))

def expected_td_target(q,dist,world,cfg):
    target=0.0
    goal_id=sid(world["goal"],world["w"])
    for ns,p in dist.items():
        done=(int(ns)==goal_id)
        r=1.0 if done else -0.01
        target += p*(r if done else r+cfg["gamma"]*float(np.max(q[int(ns)])))
    return target

def push_priority(pair,q,model,mask,wrong_map,world,cfg,heap,versions,counter):
    dist=empirical_distribution(pair,model,mask,wrong_map)
    if not dist: return counter
    si,a=pair
    priority=abs(expected_td_target(q,dist,world,cfg)-float(q[si,a]))
    if priority > cfg["priority_threshold"]:
        gen=versions.get(pair,0)+1; versions[pair]=gen
        heapq.heappush(heap,(-priority,counter,gen,pair)); counter+=1
    return counter

def run_one(world_name,world,seed,slip,bias,algorithm,cfg,out):
    w=world["w"]; n_states=world["h"]*w; n_valid=len(world["valid"])
    order,wrong_map=build_corruption_design(world,seed)
    n_corrupt=round(len(order)*bias)
    mask=set(order[:n_corrupt])
    # Algorithm-specific action randomization; map/seed gives a common corruption mask.
    agent_rng=random.Random(seed+110003+sum(ord(c) for c in algorithm)+int(slip*10000)*17)
    env_rng=random.Random(seed+210007+sum(ord(c) for c in world_name)+int(slip*10000)*19)
    eval_env_rng=random.Random(seed+310009+sum(ord(c) for c in world_name)+int(slip*10000)*23)
    eval_action_rng=random.Random(seed+410011+sum(ord(c) for c in world_name)+int(slip*10000)*29)
    env=GridWorld(world,env_rng,slip); eval_env=GridWorld(world,eval_env_rng,slip)
    q=np.zeros((n_states,4)); model={}; experienced=[]; seen=set(); visits={}
    uses_reliability=algorithm in ("dyna_q_reliability","dyna_q_no_weight","dyna_q_shuffled_reliability")
    adaptive=algorithm=="dyna_q_adaptive"
    planning=algorithm!="q_learning"
    prioritized=algorithm=="dyna_q_prioritized"
    rel_stats={}; weight_cache={}; prediction_audit={}
    episode_rows=[]; eval_rows=[]
    heap=[]; versions={}; heap_counter=0; predecessors={}
    goal_id=sid(world["goal"],w)
    for ep in range(1,cfg["episodes"]+1):
        s=env.reset(); steps=0; ret=0.0; success=0; epss=[]; applied_weights=[]; plan_updates=0
        while steps < cfg["max_steps_per_episode"]:
            si=sid(s,w); a,eps=choose(q,si,agent_rng,cfg,adaptive); epss.append(eps)
            pair=(si,a)
            # Forecast from the current learned distribution, before observing the real transition.
            forecast=empirical_distribution(pair,model,mask,wrong_map) if model.get(pair) else None
            ns,r,done=env.step(a); ni=sid(ns,w)
            q_update(q,si,a,r,ni,done,cfg,weight=1.0)  # real transition update is never weighted
            visits[pair]=visits.get(pair,0)+1
            if pair not in seen:
                seen.add(pair); experienced.append(pair)
            if uses_reliability and forecast is not None:
                loss=nll_from_forecast(forecast,ni,n_valid)
                st=rel_stats.setdefault(pair,{"n":0,"loss_sum":0.0,"zero_probability":0})
                st["n"]+=1; st["loss_sum"]+=loss
                if forecast.get(ni,0.0)<=0.0: st["zero_probability"]+=1
                weight_cache[pair]=reliability_weight(st,n_valid)
                aud=prediction_audit.setdefault(pair,{"n_predictions":0,"loss_sum":0.0,"zero_probability":0})
                aud["n_predictions"]+=1; aud["loss_sum"]+=loss
                if forecast.get(ni,0.0)<=0.0: aud["zero_probability"]+=1
            model.setdefault(pair,Counter())[ni]+=1
            if prioritized:
                dcur=empirical_distribution(pair,model,mask,wrong_map)
                for suc in dcur or {}:
                    predecessors.setdefault(int(suc),set()).add(pair)
            if planning and experienced:
                if prioritized:
                    heap_counter=push_priority(pair,q,model,mask,wrong_map,world,cfg,heap,versions,heap_counter)
                    done_updates=0
                    while done_updates < cfg["planning_steps"] and heap:
                        negpri,_,gen,(ps,pa)=heapq.heappop(heap)
                        if versions.get((ps,pa)) != gen: continue
                        versions[(ps,pa)]=gen+1
                        dist=empirical_distribution((ps,pa),model,mask,wrong_map)
                        if not dist: continue
                        target=expected_td_target(q,dist,world,cfg)
                        q[ps,pa] += cfg["alpha"]*(target-q[ps,pa])
                        plan_updates+=1; done_updates+=1
                        for pred_pair in predecessors.get(ps,set()):
                            heap_counter=push_priority(pred_pair,q,model,mask,wrong_map,world,cfg,heap,versions,heap_counter)
                else:
                    for _ in range(cfg["planning_steps"]):
                        ps,pa=agent_rng.choice(experienced)
                        dist=empirical_distribution((ps,pa),model,mask,wrong_map)
                        if not dist: continue
                        pni=sample_distribution(dist,agent_rng)
                        pdone=(pni==goal_id); pr=1.0 if pdone else -0.01
                        weight=1.0
                        if uses_reliability:
                            rel=weight_cache.get((ps,pa),0.0)
                            if algorithm=="dyna_q_no_weight": weight=1.0
                            elif algorithm=="dyna_q_shuffled_reliability":
                                keys=[k for k in weight_cache if k!=(ps,pa)]
                                weight=weight_cache[agent_rng.choice(keys)] if keys else 0.0
                            else: weight=rel
                        q_update(q,ps,pa,pr,pni,pdone,cfg,weight=weight)
                        applied_weights.append(weight); plan_updates+=1
            ret+=r; steps+=1; s=ns
            if done: success=1; break
        episode_rows.append({"maze":world_name,"grid_size":f"{world['h']}x{world['w']}",
          "valid_states":n_valid,"shortest_path_length_moves":world["shortest"],"slip_probability":slip,
          "bias_fraction":bias,"algorithm":algorithm,"seed":seed,"episode":ep,"steps":steps,
          "success":success,"extrinsic_return":ret,"mean_epsilon":float(np.mean(epss)) if epss else None,
          "planning_updates_applied":plan_updates,"mean_planning_weight_applied":float(np.mean(applied_weights)) if applied_weights else None,
          "observed_pairs":len(seen),"corrupted_pairs_predefined":len(mask),"corrupted_pairs_observed":len(mask & seen)})
    # Stochastic greedy evaluation, no exploration but same slip dynamics.
    for e in range(1,cfg["evaluation_episodes"]+1):
        s=eval_env.reset(); steps=0; ret=0.0; success=0
        while steps<cfg["max_steps_per_episode"]:
            si=sid(s,w); a,_=choose(q,si,eval_action_rng,cfg,False,greedy=True)
            ns,r,done=eval_env.step(a); ret+=r; steps+=1; s=ns
            if done: success=1; break
        eval_rows.append({"maze":world_name,"grid_size":f"{world['h']}x{world['w']}",
          "valid_states":n_valid,"shortest_path_length_moves":world["shortest"],"slip_probability":slip,
          "bias_fraction":bias,"algorithm":algorithm,"seed":seed,"evaluation_episode":e,
          "steps":steps,"success":success,"extrinsic_return":ret})
    art=out/"artifacts"/world_name/f"slip_{slip:.2f}"/f"bias_{bias:.2f}"/algorithm/f"seed_{seed:04d}"
    art.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(art/"q_table.npz",q_table=q,action_names=np.array(ACTION_NAMES))
    model_rows=[]
    for (si,a),counts in sorted(model.items()):
        pp=empirical_distribution((si,a),model,mask,wrong_map)
        model_rows.append({"state_id":si,"state":list(state(si,w)),"action":ACTION_NAMES[a],
          "observed_next_state_counts":{str(k):int(v) for k,v in sorted(counts.items())},
          "planning_distribution":{str(k):float(v) for k,v in sorted((pp or {}).items())},
          "visit_count":int(sum(counts.values())),"corrupted_prediction":bool((si,a) in mask)})
    (art/"learned_model_summary.json").write_text(json.dumps(model_rows,separators=(",",":")),encoding="utf-8")
    # Aggregate complete audit history for every pair without writing millions of redundant event rows.
    reliability_audit={}
    for pair,aud in prediction_audit.items():
        st=rel_stats[pair]
        reliability_audit[f"{pair[0]}:{pair[1]}"]={"state_id":pair[0],"action":ACTION_NAMES[pair[1]],
          "n_predictions":int(aud["n_predictions"]),"mean_prequential_nll":aud["loss_sum"]/aud["n_predictions"],
          "zero_probability_predictions":int(aud["zero_probability"]),"final_weight":reliability_weight(st,n_valid)}
    (art/"reliability_audit.json").write_text(json.dumps(reliability_audit,separators=(",",":")),encoding="utf-8")
    meta={"maze":world_name,"seed":seed,"slip_probability":slip,"bias_fraction":bias,"algorithm":algorithm,
      "config":cfg,"grid_size":f"{world['h']}x{world['w']}","valid_states":n_valid,
      "shortest_path_length_moves":world["shortest"],
      "planning_method":"expected empirical-model TD error with predecessor queue" if prioritized else "sampled transition from empirical distribution",
      "prediction_audit_count":sum(x["n_predictions"] for x in prediction_audit.values()),
      "prediction_audit_all_pre_observation":True,
      "reliability_formula":"weight = n/(n+2) * clip(1 - mean_capped_prequential_NLL/log(number_of_valid_states), 0, 1); no scored forecasts => zero",
      "corruption_scope":"selected state-action pairs have their planning next-state distribution replaced by a nonlocal non-goal distractor; real transitions unchanged",
      "information_boundary":"reliability score uses prior planning predictive distribution and subsequently observed next state; no mask labels or true transition function are passed to estimator"}
    (art/"run_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    return episode_rows,eval_rows

def append_csv(path,rows):
    if not rows: return
    exists=path.exists()
    with path.open("a",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]))
        if not exists: wr.writeheader()
        wr.writerows(rows)

def write_csv(path,rows):
    if not rows: return
    with path.open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)

def mean_sd(vals):
    vals=[float(v) for v in vals if v is not None and not (isinstance(v,float) and math.isnan(v))]
    return (statistics.mean(vals),statistics.stdev(vals) if len(vals)>1 else 0.0) if vals else (None,None)

def holm_adjust(pvals):
    m=len(pvals); order=np.argsort(pvals); out=np.ones(m); running=0.0
    for rank,idx in enumerate(order):
        adj=min(1.0,(m-rank)*float(pvals[idx])); running=max(running,adj); out[idx]=running
    return out

def main():
    parser=argparse.ArgumentParser(description="Probability-aware reliability-weighted Dyna-Q under stochastic movement")
    parser.add_argument("--seeds",type=int,default=10); parser.add_argument("--episodes",type=int,default=200)
    parser.add_argument("--max-steps",type=int,default=250); parser.add_argument("--planning-steps",type=int,default=5)
    parser.add_argument("--eval-episodes",type=int,default=10); parser.add_argument("--output",default="results_stochastic")
    parser.add_argument("--slip-levels",default="0.10,0.25"); parser.add_argument("--bias-levels",default="0,0.10,0.30")
    parser.add_argument("--priority-threshold",type=float,default=1e-4); parser.add_argument("--bootstrap-resamples",type=int,default=30000)
    parser.add_argument("--bootstrap-seed",type=int,default=20261018)
    args=parser.parse_args()
    if min(args.seeds,args.episodes,args.max_steps,args.eval_episodes)<1 or args.planning_steps<0 or args.priority_threshold<0 or args.bootstrap_resamples<100:
        parser.error("counts must be positive; planning steps/priority threshold may be zero; bootstrap >= 100")
    slips=sorted(set(float(x.strip()) for x in args.slip_levels.split(",")))
    biases=sorted(set(float(x.strip()) for x in args.bias_levels.split(",")))
    if not slips or any(x<=0 or x>=0.5 for x in slips): parser.error("slip levels must be unique values in (0, 0.5)")
    if not biases or any(x<0 or x>0.8 for x in biases): parser.error("bias levels must be in [0,0.8]")
    cfg={"episodes":args.episodes,"max_steps_per_episode":args.max_steps,"evaluation_episodes":args.eval_episodes,
      "alpha":0.1,"gamma":0.95,"epsilon_fixed":0.2,"epsilon_min":0.05,"epsilon_max":0.8,
      "lambda_dispersion":5.0,"planning_steps":args.planning_steps,"priority_threshold":args.priority_threshold,
      "step_penalty":-0.01,"goal_reward":1.0,"slip_kernel":"intended action 1-slip; each perpendicular action slip/2",
      "model":"empirical categorical distribution over previously observed next states",
      "reliability":"capped prequential NLL normalized by log(valid states), with n/(n+2) evidence shrinkage; empirical weight, not calibrated probability",
      "model_bias_scope":"selected state-action planning distributions replaced by a nonlocal non-goal distractor; real-transition learning is unmodified"}
    algs=["q_learning","dyna_q_fixed","dyna_q_adaptive","dyna_q_reliability","dyna_q_no_weight","dyna_q_shuffled_reliability","dyna_q_prioritized"]
    script=Path(__file__)
    source_sha256=hashlib.sha256(script.read_bytes()).hexdigest()
    run_id=datetime.now().strftime("stochastic_validation_%Y%m%d_%H%M%S")
    out=Path(args.output)/run_id; out.mkdir(parents=True,exist_ok=False)
    total=len(WORLD)*len(slips)*len(biases)*len(algs)*args.seeds
    config={"created_utc":datetime.now(timezone.utc).isoformat(),"run_id":run_id,"cli_args":vars(args),"config":cfg,
      "slip_levels":slips,"bias_levels":biases,"algorithms":algs,"planned_runs":total,
      "seed_values":list(range(args.seeds)),"python_version":platform.python_version(),"numpy_version":np.__version__,
      "source_sha256":source_sha256,"environment_scope":"two fixed deterministic layouts with stochastic action-slip dynamics",
      "mazes":{k:{"rows":v["rows"],"start":v["start"],"goal":v["goal"],"valid_states":len(v["valid"]),"shortest_path_length_moves":v["shortest"]} for k,v in WORLD.items()},
      "guardrails":["The reliability estimator never receives corruption-mask labels or the true transition function.",
        "The stochastic prediction is an empirical distribution over previously observed next states.",
        "The predictive log-score is a heuristic reliability weight and is not claimed calibrated.",
        "Confidence intervals condition on the two fixed maze layouts; they do not quantify generalization over all environments."]}
    (out/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    episode_path=out/"per_episode_results.csv"; eval_path=out/"greedy_evaluation_results.csv"
    episode_rows=[]; eval_rows=[]; completed=0; t0=time.time()
    for maze_name,world in WORLD.items():
      for slip in slips:
        for bias in biases:
          for alg in algs:
            for seed in range(args.seeds):
              print(f"[{completed+1}/{total}] maze={maze_name} slip={slip:.2f} bias={bias:.2f} algorithm={alg} seed={seed}",flush=True)
              er,vr=run_one(maze_name,world,seed,slip,bias,alg,cfg,out)
              episode_rows.extend(er); eval_rows.extend(vr)
              # Checkpoint each completed condition, so interruption does not lose aggregate rows.
              append_csv(episode_path,er); append_csv(eval_path,vr)
              completed+=1
    tail_n=max(1,int(math.ceil(args.episodes*0.2))); tail_start=max(1,args.episodes-tail_n+1)
    seed_rows=[]
    for maze_name in WORLD:
      for slip in slips:
       for bias in biases:
        for alg in algs:
         for seed in range(args.seeds):
          rr=[x for x in episode_rows if x["maze"]==maze_name and x["slip_probability"]==slip and x["bias_fraction"]==bias and x["algorithm"]==alg and x["seed"]==seed and x["episode"]>=tail_start]
          vv=[x for x in eval_rows if x["maze"]==maze_name and x["slip_probability"]==slip and x["bias_fraction"]==bias and x["algorithm"]==alg and x["seed"]==seed]
          seed_rows.append({"maze":maze_name,"grid_size":f"{WORLD[maze_name]['h']}x{WORLD[maze_name]['w']}",
            "valid_states":len(WORLD[maze_name]["valid"]),"shortest_path_length_moves":WORLD[maze_name]["shortest"],
            "slip_probability":slip,"bias_fraction":bias,"algorithm":alg,"seed":seed,
            "tail_steps_mean":statistics.mean(x["steps"] for x in rr),"tail_success_rate":statistics.mean(x["success"] for x in rr),
            "tail_extrinsic_return_mean":statistics.mean(x["extrinsic_return"] for x in rr),
            "eval_success_rate":statistics.mean(x["success"] for x in vv),
            "eval_steps_mean":statistics.mean(x["steps"] for x in vv),
            "eval_successful_steps_mean":statistics.mean(x["steps"] for x in vv if x["success"]) if any(x["success"] for x in vv) else None})
    write_csv(out/"seed_level_summary.csv",seed_rows)
    summary=[]
    for maze_name in WORLD:
     for slip in slips:
      for bias in biases:
       for alg in algs:
        rr=[x for x in seed_rows if x["maze"]==maze_name and x["slip_probability"]==slip and x["bias_fraction"]==bias and x["algorithm"]==alg]
        row={"maze":maze_name,"slip_probability":slip,"bias_fraction":bias,"algorithm":alg,"n_seeds":args.seeds}
        for metric in ["tail_steps_mean","tail_success_rate","tail_extrinsic_return_mean","eval_success_rate","eval_steps_mean"]:
         m,sd=mean_sd([x[metric] for x in rr]); row[metric+"_mean"]=m; row[metric+"_sd_across_seeds"]=sd
        summary.append(row)
    write_csv(out/"summary_by_maze.csv",summary)
    pooled=[]
    for slip in slips:
     for bias in biases:
      for alg in algs:
       rr=[x for x in seed_rows if x["slip_probability"]==slip and x["bias_fraction"]==bias and x["algorithm"]==alg]
       row={"slip_probability":slip,"bias_fraction":bias,"algorithm":alg,"n_maze_seed_units":len(rr)}
       for metric in ["tail_steps_mean","tail_success_rate","tail_extrinsic_return_mean","eval_success_rate","eval_steps_mean"]:
        m,sd=mean_sd([x[metric] for x in rr]); row[metric+"_mean"]=m; row[metric+"_sd_across_maze_seed_units"]=sd
       pooled.append(row)
    write_csv(out/"pooled_summary.csv",pooled)
    boot=[]; tests=[]; rng=np.random.default_rng(args.bootstrap_seed); ref="dyna_q_reliability"
    comps=[x for x in algs if x!=ref]
    for slip in slips:
     for bias in biases:
      rr=[x for x in seed_rows if x["slip_probability"]==slip and x["bias_fraction"]==bias]
      maps={(x["maze"],int(x["seed"]),x["algorithm"]):float(x["eval_success_rate"]) for x in rr}
      units=sorted({(x["maze"],int(x["seed"])) for x in rr})
      sample_idx=rng.integers(0,len(units),size=(args.bootstrap_resamples,len(units)))
      for comp in comps:
       diffs=np.array([maps[(m,s,ref)]-maps[(m,s,comp)] for m,s in units])
       bootmean=diffs[sample_idx].mean(axis=1)
       pval=1.0 if np.allclose(diffs,0) else float(wilcoxon(diffs,alternative="two-sided",zero_method="wilcox",method="auto").pvalue)
       boot.append({"slip_probability":slip,"bias_fraction":bias,"n_matched_maze_seed_units":len(units),
          "reference_algorithm":ref,"comparison_algorithm":comp,"difference_eval_success_pp_mean":100*float(diffs.mean()),
          "ci95_low_pp":100*float(np.quantile(bootmean,0.025)),"ci95_high_pp":100*float(np.quantile(bootmean,0.975)),
          "bootstrap_resamples":args.bootstrap_resamples,"unit_definition":"one maze-seed evaluation-success rate averaged over eval episodes"})
       tests.append({"slip_probability":slip,"bias_fraction":bias,"reference_algorithm":ref,"comparison_algorithm":comp,
          "mean_difference_pp":100*float(diffs.mean()),"wilcoxon_p_raw":pval})
    adj=holm_adjust([x["wilcoxon_p_raw"] for x in tests])
    for x,p in zip(tests,adj): x["wilcoxon_p_holm_all_conditions"]=float(p)
    write_csv(out/"paired_bootstrap_vs_reliability.csv",boot); write_csv(out/"paired_tests_vs_reliability.csv",tests)
    for slip in slips:
     for bias in biases:
      fig,ax=plt.subplots(figsize=(8.4,5.4))
      for alg in algs:
       vals=[x["eval_success_rate"] for x in seed_rows if x["slip_probability"]==slip and x["bias_fraction"]==bias and x["algorithm"]==alg]
       m,sd=mean_sd(vals); ax.errorbar([alg],[m],yerr=[sd],marker="o",capsize=3,label=alg)
      ax.set_ylim(-0.03,1.03); ax.set_ylabel("Greedy evaluation success rate"); ax.set_title(f"Slip={slip:.2f}, pair corruption={bias:.0%}")
      ax.tick_params(axis="x",rotation=35); ax.grid(alpha=.25); fig.tight_layout()
      fig.savefig(out/f"evaluation_success_slip_{slip:.2f}_bias_{bias:.2f}.png",dpi=160); plt.close(fig)
    (out/"run_complete.json").write_text(json.dumps({"status":"completed","completed_runs":completed,"planned_runs":total,
      "elapsed_seconds":round(time.time()-t0,2),"source_sha256":source_sha256,"files":["config.json","per_episode_results.csv",
      "greedy_evaluation_results.csv","seed_level_summary.csv","summary_by_maze.csv","pooled_summary.csv",
      "paired_bootstrap_vs_reliability.csv","paired_tests_vs_reliability.csv","evaluation_success_*.png",
      "artifacts/<maze>/slip_<p>/bias_<b>/<algorithm>/seed_<seed>/{q_table.npz,learned_model_summary.json,reliability_audit.json,run_metadata.json}"]},indent=2),encoding="utf-8")
    print("Completed.")
    print("Output:",out.resolve())

if __name__=="__main__": main()
