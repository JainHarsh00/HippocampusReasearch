#!/usr/bin/env python3
"""Diagnostic variant: explicit epsilon exploration + return-first root action."""
import csv, json, random, statistics, hashlib, platform
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from run_stochastic_reliability import parse_maze, GridWorld, ACTIONS, ACTION_NAMES, sid, q_update, build_corruption_design, empirical_distribution, sample_distribution
from run_mcts_baseline import choose_mcts_action

WORLD={"easy_8x8_open":parse_maze(["S.......","........","........","........","........","........","........",".......G"])}

def choose_return_first(root,q,model,mask,wrong_map,world,cfg,rng):
    # Same tree search; choose root action by estimated mean return first, visits as tie-break.
    root_id=sid(root,world['w']); goal_id=sid(world['goal'],world['w'])
    from collections import defaultdict
    import math
    n_sa=defaultdict(int); value_sum=defaultdict(float); n_s=defaultdict(int)
    for _ in range(cfg['mcts_simulations']):
        cur=root_id; path=[]; rewards=[]; terminal=False
        for depth in range(cfg['search_depth']):
            actions=list(ACTIONS); unvisited=[a for a in actions if n_sa[(cur,a)]==0]
            if unvisited: a=rng.choice(unvisited)
            else:
                logn=math.log(n_s[cur]+1.0)
                scores=[(value_sum[(cur,a)]/n_sa[(cur,a)]+cfg['uct_exploration']*math.sqrt(logn/n_sa[(cur,a)]),a) for a in actions]
                best=max(v for v,a in scores); a=rng.choice([a for v,a in scores if abs(v-best)<1e-12])
            path.append((cur,a)); dist=empirical_distribution((cur,a),model,mask,wrong_map)
            if not dist:
                rewards.append(cfg['step_penalty']); terminal=False; break
            nxt=sample_distribution(dist,rng); done=(nxt==goal_id)
            rewards.append(cfg['goal_reward'] if done else cfg['step_penalty'])
            if done: terminal=True; cur=nxt; break
            cur=nxt
        ret=0.0 if terminal else float(np.max(q[cur]))
        for (st,act),rew in zip(reversed(path),reversed(rewards)):
            ret=rew+cfg['gamma']*ret; key=(st,act); n_sa[key]+=1; value_sum[key]+=ret; n_s[st]+=1
    means=[value_sum[(root_id,a)]/n_sa[(root_id,a)] if n_sa[(root_id,a)] else -1e9 for a in range(4)]
    best=max(means); candidates=[a for a,v in enumerate(means) if abs(v-best)<1e-12]
    if len(candidates)>1:
        max_n=max(n_sa[(root_id,a)] for a in candidates); candidates=[a for a in candidates if n_sa[(root_id,a)]==max_n]
    return int(rng.choice(candidates)),cfg['mcts_simulations']

def run_variant(root_rule, seeds=3, episodes=60, max_steps=80, eval_episodes=5, epsilon=.2):
    world=WORLD['easy_8x8_open']; out=Path('/mnt/data/mcts_debug/corrected_diagnostic_'+root_rule); out.mkdir(parents=True,exist_ok=True)
    cfg={'episodes':episodes,'max_steps_per_episode':max_steps,'evaluation_episodes':eval_episodes,'alpha':.1,'gamma':.95,'step_penalty':-.01,'goal_reward':1.0,'mcts_simulations':8,'search_depth':12,'uct_exploration':1.414}
    rows=[]; evals=[]
    for seed in range(seeds):
        order,wrong_map=build_corruption_design(world,seed); mask=set(order[:0])
        rng=random.Random(seed+9001); env=GridWorld(world,random.Random(seed+1001),0.0); ev=GridWorld(world,random.Random(seed+2001),0.0)
        q=np.zeros((world['h']*world['w'],4)); model={}
        choose=choose_return_first if root_rule=='return_first' else choose_mcts_action
        for ep in range(1,episodes+1):
            s=env.reset(); success=0; steps=0; ret=0
            while steps<max_steps:
                # Explicit exploration is in real action selection, separate from search internals.
                if rng.random()<epsilon: a=rng.randrange(4)
                else: a,_=choose(s,q,model,mask,wrong_map,world,cfg,rng)
                si=sid(s,world['w']); ns,r,done=env.step(a); ni=sid(ns,world['w'])
                q_update(q,si,a,r,ni,done,cfg,weight=1.0); model.setdefault((si,a),{}); model[(si,a)][ni]=model[(si,a)].get(ni,0)+1
                ret+=r; steps+=1; s=ns
                if done: success=1; break
            rows.append({'root_rule':root_rule,'seed':seed,'episode':ep,'steps':steps,'success':success,'return':ret,'observed_pairs':len(model)})
        for ei in range(eval_episodes):
            s=ev.reset(); steps=0; success=0; ret=0
            while steps<max_steps:
                a,_=choose(s,q,model,mask,wrong_map,world,cfg,rng); ns,r,done=ev.step(a); s=ns; steps+=1; ret+=r
                if done: success=1; break
            evals.append({'root_rule':root_rule,'seed':seed,'evaluation_episode':ei,'steps':steps,'success':success,'return':ret})
    for name,data in [('training.csv',rows),('evaluation.csv',evals)]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=data[0].keys()); w.writeheader(); w.writerows(data)
    tail=[r for r in rows if r['episode']>episodes*.8]
    summary={'root_rule':root_rule,'seeds':seeds,'episodes_per_seed':episodes,'epsilon':epsilon,'simulations':8,'depth':12,
      'training_success_rate_tail':statistics.mean(r['success'] for r in tail),'evaluation_success_rate':statistics.mean(r['success'] for r in evals),
      'evaluation_mean_steps':statistics.mean(r['steps'] for r in evals),'python_version':platform.python_version(),'numpy_version':np.__version__,
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'note':'Diagnostic only; epsilon exploration changes the original algorithm and this is not a canonical MCTS baseline.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    summaries=[run_variant('visit_first'),run_variant('return_first')]
    print(json.dumps(summaries,indent=2))
