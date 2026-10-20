#!/usr/bin/env python3
"""Discrete-state adaptations of RND, ICM, and NGU-style intrinsic rewards.

These are explicitly lightweight adaptations for fully observed grid states. They are
not claimed to reproduce the original Atari/deep-RL implementations exactly.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, platform, random, statistics, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from run_stochastic_reliability import WORLD, GridWorld, ACTION_NAMES, choose, q_update

TORCH_DEVICE = torch.device("cpu")

def onehot(s, n):
    x = torch.zeros(n, dtype=torch.float32, device=TORCH_DEVICE)
    x[int(s)] = 1.0
    return x

class MLP(nn.Module):
    def __init__(self, n_in, n_out, width=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in,width), nn.ReLU(), nn.Linear(width,width), nn.ReLU(), nn.Linear(width,n_out))
    def forward(self, x):
        return self.net(x)

class RunningRMS:
    def __init__(self, decay=0.99): self.decay=decay; self.mean_sq=1.0
    def scale(self, x):
        self.mean_sq=self.decay*self.mean_sq+(1-self.decay)*(float(x)**2)
        return float(np.clip(float(x)/math.sqrt(max(self.mean_sq,1e-8)),0.0,5.0))

class CuriosityModule:
    def __init__(self, kind, n_states, seed, train_every=4):
        self.kind=kind; self.n_states=n_states; self.train_every=train_every; self.updates=0
        torch.manual_seed(int(seed)+851)
        self.optimizer=None; self.rms=RunningRMS()
        if kind in ("rnd","ngu_style"):
            self.target=MLP(n_states,32).to(TORCH_DEVICE)
            self.predictor=MLP(n_states,32).to(TORCH_DEVICE)
            for p in self.target.parameters(): p.requires_grad_(False)
            self.optimizer=torch.optim.Adam(self.predictor.parameters(),lr=1e-3)
        if kind=="icm":
            self.encoder=MLP(n_states,32).to(TORCH_DEVICE)
            self.inverse=MLP(64,4).to(TORCH_DEVICE)
            self.forward_model=MLP(36,32).to(TORCH_DEVICE)
            self.optimizer=torch.optim.Adam(list(self.encoder.parameters())+list(self.inverse.parameters())+list(self.forward_model.parameters()),lr=1e-3)
        if kind=="ngu_style":
            # Fixed projection approximates a compact controllable-state embedding;
            # the episodic novelty component is kept separate from lifelong RND error.
            self.episodic_embed=MLP(n_states,16).to(TORCH_DEVICE)
            for p in self.episodic_embed.parameters(): p.requires_grad_(False)

    def rnd_error(self, ns):
        x=onehot(ns,self.n_states).unsqueeze(0)
        with torch.no_grad():
            target=self.target(x); pred=self.predictor(x)
            return float(F.mse_loss(pred,target).item())

    def bonus_and_train(self, s, a, ns, episode_memory=None):
        """Return raw and scaled bonus, then perform online update when scheduled."""
        if self.kind=="rnd":
            raw=self.rnd_error(ns)
            bonus=self.rms.scale(raw)
            self.updates+=1
            if self.updates % self.train_every==0:
                x=onehot(ns,self.n_states).unsqueeze(0)
                loss=F.mse_loss(self.predictor(x),self.target(x).detach())
                self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
            return raw,bonus
        if self.kind=="icm":
            xs=onehot(s,self.n_states).unsqueeze(0); xn=onehot(ns,self.n_states).unsqueeze(0)
            aa=F.one_hot(torch.tensor([int(a)],device=TORCH_DEVICE),num_classes=4).float()
            with torch.no_grad():
                fs=self.encoder(xs); fn=self.encoder(xn); pred=self.forward_model(torch.cat([fs,aa],dim=-1))
                raw=float(F.mse_loss(pred,fn).item())
            bonus=self.rms.scale(raw)
            self.updates+=1
            if self.updates % self.train_every==0:
                fs=self.encoder(xs); fn=self.encoder(xn)
                inverse_logits=self.inverse(torch.cat([fs,fn],dim=-1))
                pred_fn=self.forward_model(torch.cat([fs,aa],dim=-1))
                inverse_loss=F.cross_entropy(inverse_logits,torch.tensor([int(a)],device=TORCH_DEVICE))
                forward_loss=F.mse_loss(pred_fn,fn.detach())
                loss=0.8*inverse_loss+0.2*forward_loss
                self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
            return raw,bonus
        if self.kind=="ngu_style":
            raw=self.rnd_error(ns)
            lifetime=self.rms.scale(raw)
            # Discrete-state episodic adaptation: revisiting a state reduces episodic
            # novelty; random projected embeddings distinguish previously unseen cells.
            counts=episode_memory if episode_memory is not None else {}
            x=onehot(ns,self.n_states).unsqueeze(0)
            with torch.no_grad(): emb=self.episodic_embed(x).squeeze(0).cpu().numpy()
            previous=counts.get("embeddings",[])
            if previous:
                d=float(min(np.linalg.norm(emb-p) for p in previous))
                epi=float(np.clip(d/(math.sqrt(16)*1.5),0.0,1.0))
            else:
                epi=1.0
            counts.setdefault("embeddings",[]).append(emb)
            counts["state_counts"][int(ns)]=counts["state_counts"].get(int(ns),0)+1
            lifelong=float(np.clip(1.0+lifetime,1.0,5.0))
            combined=epi*lifelong
            self.updates+=1
            if self.updates % self.train_every==0:
                x=onehot(ns,self.n_states).unsqueeze(0)
                loss=F.mse_loss(self.predictor(x),self.target(x).detach())
                self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
            return raw,combined
        return 0.0,0.0

def run_one(maze_name,world,slip,seed,algorithm,cfg,out):
    random.seed(seed+sum(map(ord,algorithm))*17); np.random.seed(seed+707+sum(map(ord,algorithm)))
    torch.manual_seed(seed+911+sum(map(ord,algorithm)))
    agent_rng=random.Random(seed+110003+sum(map(ord,algorithm))+int(slip*10000)*17)
    env_rng=random.Random(seed+210007+sum(map(ord,maze_name))+int(slip*10000)*19)
    eval_env_rng=random.Random(seed+310009+sum(map(ord,maze_name))+int(slip*10000)*23)
    eval_action_rng=random.Random(seed+410011+sum(map(ord,maze_name))+int(slip*10000)*29)
    n_states=world["h"]*world["w"]
    env=GridWorld(world,env_rng,slip); eval_env=GridWorld(world,eval_env_rng,slip)
    q=np.zeros((n_states,4)); visits={}; curiosity=None
    if algorithm in ("rnd_q","icm_q","ngu_style_q"):
        kind={"rnd_q":"rnd","icm_q":"icm","ngu_style_q":"ngu_style"}[algorithm]
        curiosity=CuriosityModule(kind,n_states,seed,cfg["curiosity_train_every"])
    ep_rows=[]; eval_rows=[]; total_steps=0
    for ep in range(1,cfg["episodes"]+1):
        s=env.reset(); steps=0; ret=0.; intrinsic_sum=0.; success=0; eps=[]
        episode_memory={"embeddings":[],"state_counts":{}}
        while steps<cfg["max_steps_per_episode"]:
            si=s[0]*world["w"]+s[1]
            a,epv=choose(q,si,agent_rng,cfg,False); eps.append(epv)
            ns,r,done=env.step(a); ni=ns[0]*world["w"]+ns[1]
            pair=(si,a); visits[pair]=visits.get(pair,0)+1
            intrinsic=0.
            if algorithm=="count_bonus_q":
                intrinsic=cfg["intrinsic_scale"]/math.sqrt(1.0+visits[pair])
            elif curiosity is not None:
                _,b=curiosity.bonus_and_train(si,a,ni,episode_memory if algorithm=="ngu_style_q" else None)
                intrinsic=cfg["intrinsic_scale"]*b
            q_update(q,si,a,float(r+intrinsic),ni,done,cfg,weight=1.0)
            total_steps+=1; steps+=1; ret+=r; intrinsic_sum+=intrinsic; s=ns
            if done: success=1; break
        ep_rows.append({"maze":maze_name,"grid_size":f"{world['h']}x{world['w']}","shortest_path_length_moves":world["shortest"],
          "slip_probability":slip,"bias_fraction":0.0,"algorithm":algorithm,"seed":seed,"episode":ep,"steps":steps,
          "success":success,"extrinsic_return":ret,"intrinsic_reward_sum":intrinsic_sum,
          "mean_epsilon":float(np.mean(eps)) if eps else None})
    for e in range(1,cfg["evaluation_episodes"]+1):
        s=eval_env.reset(); steps=0; ret=0.; success=0
        while steps<cfg["max_steps_per_episode"]:
            si=s[0]*world["w"]+s[1]
            a,_=choose(q,si,eval_action_rng,cfg,False,greedy=True)
            ns,r,done=eval_env.step(a); steps+=1; ret+=r; s=ns
            if done: success=1; break
        eval_rows.append({"maze":maze_name,"grid_size":f"{world['h']}x{world['w']}","shortest_path_length_moves":world["shortest"],
          "slip_probability":slip,"bias_fraction":0.0,"algorithm":algorithm,"seed":seed,"evaluation_episode":e,
          "steps":steps,"success":success,"extrinsic_return":ret})
    art=out/"artifacts"/maze_name/f"slip_{slip:.2f}"/algorithm/f"seed_{seed:04d}"; art.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(art/"q_table.npz",q_table=q,action_names=np.array(ACTION_NAMES))
    if curiosity is not None:
        modules={}
        if algorithm in ("rnd_q","ngu_style_q"):
            modules["target"]={k:v.detach().cpu().numpy() for k,v in curiosity.target.state_dict().items()}
            modules["predictor"]={k:v.detach().cpu().numpy() for k,v in curiosity.predictor.state_dict().items()}
        if algorithm=="icm_q":
            for nm in ("encoder","inverse","forward_model"):
                modules[nm]={k:v.detach().cpu().numpy() for k,v in getattr(curiosity,nm).state_dict().items()}
        np.savez_compressed(art/"intrinsic_module.npz",**{f"{m}__{k}":v for m,sd in modules.items() for k,v in sd.items()})
    meta={"maze":maze_name,"seed":seed,"slip_probability":slip,"algorithm":algorithm,"config":cfg,
      "python_version":platform.python_version(),"torch_version":torch.__version__,"device":"cpu",
      "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      "helper_source_sha256":hashlib.sha256((Path(__file__).parent/"run_stochastic_reliability.py").read_bytes()).hexdigest(),
      "algorithm_notes":{
        "q_learning":"Tabular Q-learning, extrinsic reward only.",
        "count_bonus_q":"Tabular Q-learning plus beta/sqrt(1+state-action visit count).",
        "rnd_q":"RND-style: fixed random target MLP and trained predictor MLP; intrinsic reward is normalized prediction error.",
        "icm_q":"ICM-style: learned feature encoder, inverse action predictor, forward feature predictor; intrinsic reward is forward feature prediction error.",
        "ngu_style_q":"NGU-style discrete adaptation: within-episode novelty from nearest-neighbor distance in a fixed random embedding multiplied by lifelong RND novelty. Does not implement NGU's full UVFA/distributed recurrent agent."}[algorithm],
      "curiosity_train_every":cfg["curiosity_train_every"] if curiosity else None,
      "scope":"No injected planning-model corruption: intrinsic motivation baselines are model-free and assessed on stochastic navigation sample efficiency."}
    (art/"run_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    return ep_rows,eval_rows

def write_csv(path,rows):
    if not rows: return
    with path.open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)

def mean_sd(vals):
    vals=[float(v) for v in vals if v is not None]
    return statistics.mean(vals), statistics.stdev(vals) if len(vals)>1 else 0.0

def main():
    p=argparse.ArgumentParser(description="RND/ICM/NGU-style intrinsic-reward baselines for stochastic grid navigation")
    p.add_argument("--seeds",type=int,default=10); p.add_argument("--episodes",type=int,default=200)
    p.add_argument("--max-steps",type=int,default=250); p.add_argument("--eval-episodes",type=int,default=10)
    p.add_argument("--slip-levels",default="0.10,0.25"); p.add_argument("--output",default="results_intrinsic")
    p.add_argument("--intrinsic-scale",type=float,default=0.05); p.add_argument("--curiosity-train-every",type=int,default=4)
    args=p.parse_args()
    if min(args.seeds,args.episodes,args.max_steps,args.eval_episodes,args.curiosity_train_every)<1 or args.intrinsic_scale<0:
        p.error("counts must be positive; intrinsic-scale must be nonnegative")
    slips=sorted(set(float(x.strip()) for x in args.slip_levels.split(",")))
    if not slips or any(x<=0 or x>=0.5 for x in slips): p.error("slip probabilities must be in (0,0.5)")
    cfg={"episodes":args.episodes,"max_steps_per_episode":args.max_steps,"evaluation_episodes":args.eval_episodes,
      "alpha":0.1,"gamma":0.95,"epsilon_fixed":0.2,"epsilon_min":0.05,"epsilon_max":0.8,"lambda_dispersion":5.0,
      "intrinsic_scale":args.intrinsic_scale,"curiosity_train_every":args.curiosity_train_every,
      "reward":"real reward remains +1 at goal and -0.01 otherwise; intrinsic reward only added to learning target; evaluation reports extrinsic return"}
    algs=["q_learning","count_bonus_q","rnd_q","icm_q","ngu_style_q"]
    out=Path(args.output)/datetime.now().strftime("intrinsic_baselines_%Y%m%d_%H%M%S"); out.mkdir(parents=True,exist_ok=False)
    total=len(WORLD)*len(slips)*len(algs)*args.seeds
    config={"planned_runs":total,"created_utc":datetime.now(timezone.utc).isoformat(),"algorithms":algs,"slip_levels":slips,
      "seeds":list(range(args.seeds)),"config":cfg,"python_version":platform.python_version(),"numpy_version":np.__version__,"torch_version":torch.__version__,
      "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      "helper_source_sha256":hashlib.sha256((Path(__file__).parent/"run_stochastic_reliability.py").read_bytes()).hexdigest(),
      "environment_scope":"two fixed grid layouts under stochastic action-slip dynamics; model-bias corruption is not applicable to these model-free exploration baselines",
      "notes":["RND, ICM, and NGU are implemented as discrete-state adaptations; the original methods target high-dimensional/deep-RL settings.","NGU-style variant omits original UVFA policy family and distributed recurrent training.","This suite measures exploration under stochastic navigation separately from the model-corruption robustness experiment."]}
    (out/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    eps_rows=[]; eval_rows=[]; completed=0; t0=time.time()
    for mn,world in WORLD.items():
        for slip in slips:
            for alg in algs:
                for seed in range(args.seeds):
                    completed+=1; print(f"[{completed}/{total}] maze={mn} slip={slip:.2f} algorithm={alg} seed={seed}",flush=True)
                    er,vr=run_one(mn,world,slip,seed,alg,cfg,out); eps_rows.extend(er); eval_rows.extend(vr)
                    # Write checkpoints after each condition to reduce risk of data loss.
                    write_csv(out/"per_episode_results.csv",eps_rows); write_csv(out/"greedy_evaluation_results.csv",eval_rows)
    tail_start=max(1,args.episodes-int(math.ceil(args.episodes*0.2))+1)
    seed_rows=[]
    for mn in WORLD:
        for slip in slips:
            for alg in algs:
                for seed in range(args.seeds):
                    rr=[x for x in eps_rows if x["maze"]==mn and x["slip_probability"]==slip and x["algorithm"]==alg and x["seed"]==seed and x["episode"]>=tail_start]
                    vv=[x for x in eval_rows if x["maze"]==mn and x["slip_probability"]==slip and x["algorithm"]==alg and x["seed"]==seed]
                    seed_rows.append({"maze":mn,"slip_probability":slip,"algorithm":alg,"seed":seed,
                      "tail_steps_mean":statistics.mean(x["steps"] for x in rr),"tail_success_rate":statistics.mean(x["success"] for x in rr),
                      "tail_extrinsic_return_mean":statistics.mean(x["extrinsic_return"] for x in rr),"eval_success_rate":statistics.mean(x["success"] for x in vv),
                      "eval_steps_mean":statistics.mean(x["steps"] for x in vv),"eval_successful_steps_mean":statistics.mean(x["steps"] for x in vv if x["success"]) if any(x["success"] for x in vv) else None})
    write_csv(out/"seed_level_summary.csv",seed_rows)
    pooled=[]
    for slip in slips:
        for alg in algs:
            rr=[x for x in seed_rows if x["slip_probability"]==slip and x["algorithm"]==alg]
            row={"slip_probability":slip,"algorithm":alg,"n_maze_seed_units":len(rr)}
            for met in ("tail_steps_mean","tail_success_rate","tail_extrinsic_return_mean","eval_success_rate","eval_steps_mean"):
                m,sd=mean_sd([x[met] for x in rr]); row[met+"_mean"]=m; row[met+"_sd_across_maze_seed_units"]=sd
            pooled.append(row)
    write_csv(out/"pooled_summary.csv",pooled)
    (out/"run_complete.json").write_text(json.dumps({"status":"completed","completed_runs":completed,"planned_runs":total,
      "elapsed_seconds":round(time.time()-t0,2),"source_sha256":config["source_sha256"],"files":["config.json","per_episode_results.csv","greedy_evaluation_results.csv","seed_level_summary.csv","pooled_summary.csv","artifacts/"]},indent=2),encoding="utf-8")
    print("Completed. Output:",out.resolve())

if __name__=="__main__": main()
