#!/usr/bin/env python3
"""Online UCT-MCTS planning baseline under learned-model corruption.

The method uses a learned empirical transition model (with the same planning-only
corruption intervention), UCT selection, and a tabular Q-value leaf bootstrap. Real
transitions update Q with ordinary Q-learning. This hybrid is described explicitly;
it is not claimed to be a verbatim reproduction of a single canonical MCTS-RL system.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, platform, random, statistics, time
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from run_stochastic_reliability import WORLD, GridWorld, ACTIONS, ACTION_NAMES, sid, state, q_update, build_corruption_design, empirical_distribution, sample_distribution

def choose_mcts_action(root, q, model, mask, wrong_map, world, cfg, rng):
    """UCT search rooted at the current state; backed up returns include Q bootstrap."""
    root_id=sid(root,world["w"]); goal_id=sid(world["goal"],world["w"])
    n_sa=defaultdict(int); value_sum=defaultdict(float); n_s=defaultdict(int)
    sims=cfg["mcts_simulations"]; max_depth=cfg["search_depth"]; c=cfg["uct_exploration"]
    for _ in range(sims):
        cur=root_id; path=[]; rewards=[]; terminal=False
        for depth in range(max_depth):
            actions=list(ACTIONS)
            unvisited=[a for a in actions if n_sa[(cur,a)]==0]
            if unvisited:
                a=rng.choice(unvisited)
            else:
                logn=math.log(n_s[cur]+1.0)
                scores=[]
                for aa in actions:
                    key=(cur,aa)
                    score=value_sum[key]/n_sa[key] + c*math.sqrt(logn/n_sa[key])
                    scores.append((score,aa))
                best=max(v for v,_ in scores)
                a=rng.choice([aa for v,aa in scores if abs(v-best)<1e-12])
            path.append((cur,a))
            dist=empirical_distribution((cur,a),model,mask,wrong_map)
            if not dist:
                rewards.append(cfg["step_penalty"])
                # Unknown pair terminates model rollout; leaf estimate comes from Q(s,*).
                terminal=False
                break
            nxt=sample_distribution(dist,rng)
            done=(nxt==goal_id)
            rewards.append(cfg["goal_reward"] if done else cfg["step_penalty"])
            if done:
                terminal=True; cur=nxt; break
            cur=nxt
        # Bootstrap from Q unless the simulated path reached the terminal goal.
        ret=0.0
        if not terminal:
            ret=float(np.max(q[cur]))
        for (st,act),rew in zip(reversed(path),reversed(rewards)):
            ret=rew+cfg["gamma"]*ret
            key=(st,act); n_sa[key]+=1; value_sum[key]+=ret; n_s[st]+=1
    counts=[n_sa[(root_id,a)] for a in ACTIONS]
    max_count=max(counts)
    candidates=[a for a,n in enumerate(counts) if n==max_count]
    if len(candidates)>1:
        means={a:(value_sum[(root_id,a)]/n_sa[(root_id,a)] if n_sa[(root_id,a)] else float(np.max(q[root_id]))) for a in candidates}
        best=max(means.values()); candidates=[a for a,v in means.items() if abs(v-best)<1e-12]
    return int(rng.choice(candidates)),int(sims)

def write_csv(path,rows):
    if not rows:return
    exists=path.exists()
    with path.open("a" if exists else "w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        if not exists:w.writeheader()
        w.writerows(rows)

def run_one(maze_name,world,seed,slip,bias,cfg,out):
    order,wrong_map=build_corruption_design(world,seed)
    n_corrupt=round(len(order)*bias); mask=set(order[:n_corrupt])
    agent_rng=random.Random(seed+515011+sum(map(ord,maze_name))+int(slip*10000)*11)
    env_rng=random.Random(seed+210007+sum(map(ord,maze_name))+int(slip*10000)*19)
    eval_env_rng=random.Random(seed+310009+sum(map(ord,maze_name))+int(slip*10000)*23)
    eval_planner_rng=random.Random(seed+610013+sum(map(ord,maze_name))+int(slip*10000)*31)
    n_states=world["h"]*world["w"]
    env=GridWorld(world,env_rng,slip); eval_env=GridWorld(world,eval_env_rng,slip)
    q=np.zeros((n_states,4)); model={}; visits={}; ep_rows=[]; eval_rows=[]; completed_searches=0
    goal_id=sid(world["goal"],world["w"])
    for ep in range(1,cfg["episodes"]+1):
        s=env.reset(); steps=0; ret=0.; success=0; sims_used=0
        while steps<cfg["max_steps_per_episode"]:
            a,n_sims=choose_mcts_action(s,q,model,mask,wrong_map,world,cfg,agent_rng)
            sims_used+=n_sims
            si=sid(s,world["w"]); ns,r,done=env.step(a); ni=sid(ns,world["w"])
            q_update(q,si,a,r,ni,done,cfg,weight=1.0)
            model.setdefault((si,a),Counter())[ni]+=1
            visits[(si,a)]=visits.get((si,a),0)+1
            ret+=r; steps+=1; s=ns
            if done: success=1; break
        ep_rows.append({"maze":maze_name,"grid_size":f"{world['h']}x{world['w']}","shortest_path_length_moves":world["shortest"],
          "slip_probability":slip,"bias_fraction":bias,"algorithm":"uct_mcts_q_bootstrap","seed":seed,"episode":ep,"steps":steps,
          "success":success,"extrinsic_return":ret,"mcts_simulations":sims_used,"model_pairs_observed":len(model),"corrupted_pairs_predefined":len(mask)})
    for e in range(1,cfg["evaluation_episodes"]+1):
        s=eval_env.reset(); steps=0; ret=0.; success=0; sims_used=0
        while steps<cfg["max_steps_per_episode"]:
            a,n_sims=choose_mcts_action(s,q,model,mask,wrong_map,world,cfg,eval_planner_rng); sims_used+=n_sims
            ns,r,done=eval_env.step(a); steps+=1; ret+=r; s=ns
            if done:success=1;break
        eval_rows.append({"maze":maze_name,"grid_size":f"{world['h']}x{world['w']}","shortest_path_length_moves":world["shortest"],
          "slip_probability":slip,"bias_fraction":bias,"algorithm":"uct_mcts_q_bootstrap","seed":seed,"evaluation_episode":e,
          "steps":steps,"success":success,"extrinsic_return":ret,"mcts_simulations":sims_used})
    art=out/"artifacts"/maze_name/f"slip_{slip:.2f}"/f"bias_{bias:.2f}"/f"seed_{seed:04d}"; art.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(art/"q_table.npz",q_table=q,action_names=np.array(ACTION_NAMES))
    model_rows=[]
    for (si,a),counts in sorted(model.items()):
        dist=empirical_distribution((si,a),model,mask,wrong_map)
        model_rows.append({"state_id":si,"action":ACTION_NAMES[a],"observed_next_state_counts":{str(k):int(v) for k,v in counts.items()},
          "planning_distribution":{str(k):float(v) for k,v in (dist or {}).items()},"corrupted_prediction":(si,a) in mask})
    (art/"learned_model.json").write_text(json.dumps(model_rows,separators=(",",":")),encoding="utf-8")
    meta={"maze":maze_name,"seed":seed,"slip_probability":slip,"bias_fraction":bias,"algorithm":"uct_mcts_q_bootstrap","config":cfg,
      "python_version":platform.python_version(),"numpy_version":np.__version__,"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      "helper_source_sha256":hashlib.sha256((Path(__file__).parent/"run_stochastic_reliability.py").read_bytes()).hexdigest(),
      "method":"UCT tree search over online empirical transition model; selected action maximizes root visits with mean-return tie-break; tabular Q-learning used for real transitions and as leaf bootstrap",
      "search_budget":"fixed simulations per real decision and a fixed depth cap; model-query budget is higher than five one-step Dyna updates and must be reported explicitly",
      "corruption_scope":"selected planning state-action distributions replaced by nonlocal non-goal distractor; environment observations and real updates unchanged"}
    (art/"run_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    return ep_rows,eval_rows

def main():
    p=argparse.ArgumentParser(description="UCT-MCTS planning baseline under stochastic movement and controlled planning-model corruption")
    p.add_argument("--seeds",type=int,default=10);p.add_argument("--episodes",type=int,default=200);p.add_argument("--max-steps",type=int,default=250)
    p.add_argument("--eval-episodes",type=int,default=10);p.add_argument("--slip-levels",default="0.10,0.25");p.add_argument("--bias-levels",default="0,0.10,0.30")
    p.add_argument("--mcts-simulations",type=int,default=32);p.add_argument("--search-depth",type=int,default=48);p.add_argument("--uct-exploration",type=float,default=1.414)
    p.add_argument("--output",default="results_mcts")
    args=p.parse_args()
    if min(args.seeds,args.episodes,args.max_steps,args.eval_episodes,args.mcts_simulations,args.search_depth)<1 or args.uct_exploration<0:p.error("counts must be positive; UCT exploration must be nonnegative")
    slips=sorted(set(float(x.strip()) for x in args.slip_levels.split(",")));biases=sorted(set(float(x.strip()) for x in args.bias_levels.split(",")))
    if not slips or any(x<=0 or x>=0.5 for x in slips):p.error("slip levels must be in (0,0.5)")
    if not biases or any(x<0 or x>0.8 for x in biases):p.error("bias levels must be in [0,0.8]")
    cfg={"episodes":args.episodes,"max_steps_per_episode":args.max_steps,"evaluation_episodes":args.eval_episodes,
      "alpha":0.1,"gamma":0.95,"step_penalty":-0.01,"goal_reward":1.0,"mcts_simulations":args.mcts_simulations,
      "search_depth":args.search_depth,"uct_exploration":args.uct_exploration,
      "search_horizon_note":"Default search depth 48 exceeds the longest shortest path (40 moves) in the included mazes; this removes the previous 12-step horizon limitation."}
    out=Path(args.output)/datetime.now().strftime("mcts_validation_%Y%m%d_%H%M%S");out.mkdir(parents=True,exist_ok=False)
    total=len(WORLD)*len(slips)*len(biases)*args.seeds
    src=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    meta={"created_utc":datetime.now(timezone.utc).isoformat(),"planned_runs":total,"slip_levels":slips,"bias_levels":biases,
      "seeds":list(range(args.seeds)),"config":cfg,"source_sha256":src,"helper_source_sha256":hashlib.sha256((Path(__file__).parent/"run_stochastic_reliability.py").read_bytes()).hexdigest(),"python_version":platform.python_version(),"numpy_version":np.__version__,
      "algorithm":"UCT-MCTS with learned empirical stochastic model and Q-value bootstrap","caveats":["This is an explicit hybrid baseline: real transitions update tabular Q-learning, and Q values bootstrap search leaves.","Search budget is not compute-matched to five one-step Dyna updates; report simulation/model-query count and runtime.","MCTS uses the same planning-only corruption intervention, not corrupted real transitions.","Default search depth 48 exceeds the longest 40-move shortest path, unlike the previous 12-step search horizon."]}
    (out/"config.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    ep_rows=[];eval_rows=[];completed=0;t0=time.time()
    for mn,world in WORLD.items():
      for slip in slips:
        for bias in biases:
          for seed in range(args.seeds):
            completed+=1;print(f"[{completed}/{total}] maze={mn} slip={slip:.2f} bias={bias:.2f} algorithm=uct_mcts_q_bootstrap seed={seed}",flush=True)
            er,vr=run_one(mn,world,seed,slip,bias,cfg,out);ep_rows.extend(er);eval_rows.extend(vr)
            write_csv(out/"per_episode_results.csv",er);write_csv(out/"greedy_evaluation_results.csv",vr)
    tail_start=max(1,args.episodes-int(math.ceil(args.episodes*0.2))+1);seed_rows=[]
    for mn in WORLD:
      for slip in slips:
        for bias in biases:
          for seed in range(args.seeds):
            rr=[x for x in ep_rows if x["maze"]==mn and x["slip_probability"]==slip and x["bias_fraction"]==bias and x["seed"]==seed and x["episode"]>=tail_start]
            vv=[x for x in eval_rows if x["maze"]==mn and x["slip_probability"]==slip and x["bias_fraction"]==bias and x["seed"]==seed]
            seed_rows.append({"maze":mn,"slip_probability":slip,"bias_fraction":bias,"seed":seed,
              "tail_steps_mean":statistics.mean(x["steps"] for x in rr),"tail_success_rate":statistics.mean(x["success"] for x in rr),
              "eval_success_rate":statistics.mean(x["success"] for x in vv),"eval_steps_mean":statistics.mean(x["steps"] for x in vv),
              "mean_mcts_simulations_per_episode":statistics.mean(x["mcts_simulations"] for x in rr),
              "mcts_simulations_per_environment_step":sum(x["mcts_simulations"] for x in rr)/max(1,sum(x["steps"] for x in rr))})
    write_csv(out/"seed_level_summary.csv",seed_rows)
    pooled=[]
    for slip in slips:
      for bias in biases:
        rr=[x for x in seed_rows if x["slip_probability"]==slip and x["bias_fraction"]==bias]
        row={"slip_probability":slip,"bias_fraction":bias,"n_maze_seed_units":len(rr)}
        for met in ("tail_steps_mean","tail_success_rate","eval_success_rate","eval_steps_mean","mean_mcts_simulations_per_episode","mcts_simulations_per_environment_step"):
          m,sd=statistics.mean([x[met] for x in rr]),statistics.stdev([x[met] for x in rr]) if len(rr)>1 else 0.0
          row[met+"_mean"]=m;row[met+"_sd_across_maze_seed_units"]=sd
        pooled.append(row)
    write_csv(out/"pooled_summary.csv",pooled)
    (out/"run_complete.json").write_text(json.dumps({"status":"completed","completed_runs":completed,"planned_runs":total,
      "elapsed_seconds":round(time.time()-t0,2),"source_sha256":src,"files":["config.json","per_episode_results.csv","greedy_evaluation_results.csv","seed_level_summary.csv","pooled_summary.csv","artifacts/"]},indent=2),encoding="utf-8")
    print("Completed. Output:",out.resolve())

def write_csv(path,rows):
    if not rows:return
    exists=path.exists()
    with path.open("a" if exists else "w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        if not exists:w.writeheader()
        w.writerows(rows)

if __name__=="__main__":main()
