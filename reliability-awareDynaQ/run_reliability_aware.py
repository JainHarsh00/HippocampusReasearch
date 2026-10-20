#!/usr/bin/env python3
"""Reliability-aware planning experiment for biased Dyna planning models.

Compares Q-learning, fixed-epsilon Dyna-Q, adaptive-epsilon Dyna-Q, and reliability-aware Dyna-Q over nested model-bias levels and multiple fixed mazes. Impairment masks are defined
before training, nested across levels, and independent of algorithm visitation.
"""
from __future__ import annotations
import argparse, csv, json, math, random, statistics, time
from datetime import datetime, timezone
from pathlib import Path
from collections import deque
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ACTIONS = {0:(-1,0), 1:(0,1), 2:(1,0), 3:(0,-1)}
ACTION_NAMES = ['up','right','down','left']
MAZES = {
 'maze_a': ['S...#...', '.##.#.#.', '...#....', '.#...##.', '.###.#..', '...#.#..', '##...#..', '.......G'],
 'maze_b': ['S.#.....', '...#.#..', '.#...#..', '.###....', '...##.#.', '#.....#.', '..#.....', '.....#.G'],
 'maze_c': ['S....#..', '##.#....', '...#.#..', '.#...#..', '.#.#....', '...##.#.', '.#......', '...#...G'],
}

def parse_maze(rows):
    h,w=len(rows),len(rows[0])
    if any(len(r)!=w for r in rows): raise ValueError('Maze rows must have equal width')
    start=next(((r,c) for r,row in enumerate(rows) for c,ch in enumerate(row) if ch=='S'),None)
    goal=next(((r,c) for r,row in enumerate(rows) for c,ch in enumerate(row) if ch=='G'),None)
    if start is None or goal is None: raise ValueError('Each maze needs one S and one G')
    walls={(r,c) for r,row in enumerate(rows) for c,ch in enumerate(row) if ch=='#'}
    valid=[(r,c) for r in range(h) for c in range(w) if (r,c) not in walls]
    def nxt(s,a):
        dr,dc=ACTIONS[a];n=(s[0]+dr,s[1]+dc)
        return n if 0<=n[0]<h and 0<=n[1]<w and n not in walls else s
    q=deque([(start,0)]);seen={start};dist=None
    while q:
        s,d=q.popleft()
        if s==goal:dist=d;break
        for a in ACTIONS:
            n=nxt(s,a)
            if n not in seen:seen.add(n);q.append((n,d+1))
    if dist is None: raise ValueError(f'Unsolvable maze: {rows}')
    return {'rows':rows,'h':h,'w':w,'start':start,'goal':goal,'walls':walls,'valid':valid,'nxt':nxt,'shortest':dist}

WORLD={k:parse_maze(v) for k,v in MAZES.items()}

def sid(s,w):return s[0]*w+s[1]
def state(i,w):return (i//w,i%w)

def eps_for(qrow,cfg,adaptive):
    if not adaptive:return cfg['epsilon_fixed']
    dispersion=float(np.max(qrow)-np.min(qrow))
    return float(np.clip(cfg['epsilon_min']+(cfg['epsilon_max']-cfg['epsilon_min'])*math.exp(-cfg['lambda_dispersion']*dispersion),cfg['epsilon_min'],cfg['epsilon_max']))

def choose(q,s,rng,cfg,adaptive,greedy=False):
    eps=0.0 if greedy else eps_for(q[s],cfg,adaptive)
    if rng.random()<eps:return rng.randrange(4),eps
    best=np.flatnonzero(np.isclose(q[s],np.max(q[s])))
    return int(rng.choice(best.tolist())),eps

def update(q,s,a,r,ns,done,cfg):
    target=r if done else r+cfg['gamma']*float(np.max(q[ns]))
    q[s,a]+=cfg['alpha']*(target-q[s,a])

def update_reliability_weighted(q,s,a,r,ns,done,cfg,reliability):
    target=r if done else r+cfg['gamma']*float(np.max(q[ns]))
    q[s,a]+=cfg['alpha']*float(np.clip(reliability,0.0,1.0))*(target-q[s,a])

class GridWorld:
    def __init__(self,world,rng):self.world=world;self.rng=rng;self.s=world['start']
    def reset(self):self.s=self.world['start'];return self.s
    def step(self,a):
        ns=self.world['nxt'](self.s,int(a));done=ns==self.world['goal'];self.s=ns
        return ns,(1.0 if done else -0.01),done

def build_bias_order(world,seed):
    """One pre-sampled order; bias levels are nested prefixes for paired comparisons."""
    candidates=[]
    for s in world['valid']:
        if s==world['goal']:continue
        for a in ACTIONS:
            true_ns=world['nxt'](s,a)
            if true_ns==world['goal']:continue
            candidates.append((sid(s,world['w']),a,sid(true_ns,world['w'])))
    rng=random.Random(seed*1000003 + sum(ord(c) for c in ''.join(world['rows']))*97 + 8081)
    rng.shuffle(candidates)
    non_goal=[sid(s,world['w']) for s in world['valid'] if s!=world['goal']]
    mapping={}
    for si,a,true_ni in candidates:
        opts=[x for x in non_goal if x!=true_ni]
        mapping[(si,a)]=rng.choice(opts)
    return candidates,mapping

def predict_from_learned_model(pair, model, corrupted_pairs, corrupt_map):
    """Return the model's predicted next state, or None if no observations exist."""
    if pair not in model or not model[pair]:
        return None, None
    outcomes = model[pair]
    next_counts = {}
    terminal_counts = {}
    for ni, _r, done in outcomes:
        next_counts[ni] = next_counts.get(ni, 0) + 1
        terminal_counts[(ni, bool(done))] = terminal_counts.get((ni, bool(done)), 0) + 1
    predicted_ni = sorted(next_counts, key=lambda ni: (-next_counts[ni], ni))[0]
    predicted_done = sorted(
        terminal_counts, key=lambda item: (-terminal_counts[item], item[0], item[1])
    )[0][1]
    # The experimental corruption changes the model prediction, but the reliability
    # estimator receives only the prediction and subsequent observation—not the mask.
    if pair in corrupted_pairs and not predicted_done:
        predicted_ni = corrupt_map[pair]
    return predicted_ni, predicted_done


def run_one(world_name,world,seed,bias_fraction,algorithm,cfg,out):
    w=world['w'];n_states=world['h']*w; goal_id=sid(world['goal'],w)
    order,corrupt_map=build_bias_order(world,seed)
    n_corrupt=round(len(order)*bias_fraction)
    corrupted_pairs={(si,a) for si,a,_ in order[:n_corrupt]}
    agent_rng=random.Random(seed+100003+sum(ord(c) for c in algorithm))
    env_rng=random.Random(seed+200003+sum(ord(c) for c in world_name))
    eval_env_rng=random.Random(seed+300007+sum(ord(c) for c in world_name))
    eval_action_rng=np.random.default_rng(seed+400009+sum(ord(c) for c in algorithm)+sum(ord(c) for c in world_name))
    env=GridWorld(world,env_rng);eval_env=GridWorld(world,eval_env_rng)
    q=np.zeros((n_states,4));model={};experienced=[];seen=set();visits={};episode_rows=[]
    reliability_counts={}  # pair -> [matches, observations] for predicted-vs-observed next-state agreement
    planning=algorithm!='q_learning';adaptive=algorithm=='dyna_q_adaptive'; reliability_aware=algorithm=='dyna_q_reliability'
    for ep in range(1,cfg['episodes']+1):
        s=env.reset();steps=0;ret=0.;success=0;epss=[];plan_updates=0;planning_reliabilities=[]
        while steps<cfg['max_steps_per_episode']:
            si=sid(s,w);a,eps=choose(q,si,agent_rng,cfg,adaptive);epss.append(eps)
            pair=(si,a)
            # Predict from the learned model before observing this transition.
            predicted_ni_before_observation, _ = predict_from_learned_model(
                pair, model, corrupted_pairs, corrupt_map
            )
            ns,r,done=env.step(a);ni=sid(ns,w);update(q,si,a,r,ni,done,cfg)
            visits[pair]=visits.get(pair,0)+1
            if pair not in seen:seen.add(pair);experienced.append(pair)
            if reliability_aware and predicted_ni_before_observation is not None:
                matches, observations = reliability_counts.get(pair, (0, 0))
                matches += int(predicted_ni_before_observation == ni)
                observations += 1
                reliability_counts[pair] = (matches, observations)
            model.setdefault(pair,[]).append((ni,r,done))
            if planning and experienced:
                for _ in range(cfg['planning_steps']):
                    ps,pa=agent_rng.choice(experienced)
                    _obs_ni,pr,pdone=agent_rng.choice(model[(ps,pa)])
                    pni, predicted_done = predict_from_learned_model(
                        (ps,pa), model, corrupted_pairs, corrupt_map
                    )
                    if pni is None:
                        # No evidence for this state-action pair: do not invent a prediction.
                        continue
                    pdone = bool(pdone or predicted_done)
                    if reliability_aware:
                        matches, observations = reliability_counts.get((ps,pa), (0, 0))
                        reliability = ((matches + 1.0) / (observations + 2.0)
                                       if observations else 0.0)
                        planning_reliabilities.append(reliability)
                        update_reliability_weighted(q,ps,pa,pr,pni,pdone,cfg,reliability)
                    else:
                        update(q,ps,pa,pr,pni,pdone,cfg)
                    plan_updates+=1
            ret+=r;steps+=1;s=ns
            if done:success=1;break
        episode_rows.append({'maze':world_name,'bias_fraction':bias_fraction,'algorithm':algorithm,'seed':seed,'episode':ep,'steps':steps,'success':success,'extrinsic_return':ret,'mean_epsilon':float(np.mean(epss)) if epss else None,'planning_updates_applied':plan_updates,'mean_planning_reliability':float(np.mean(planning_reliabilities)) if planning_reliabilities else None,'observed_pairs':len(seen),'corrupted_pairs_predefined':len(corrupted_pairs),'corrupted_pairs_observed':len(corrupted_pairs & seen),'corrupted_pairs_observed_fraction':len(corrupted_pairs & seen)/max(1,len(corrupted_pairs))})
    eval_rows=[]
    for e in range(1,cfg['evaluation_episodes']+1):
        s=eval_env.reset();steps=0;ret=0.;success=0
        while steps<cfg['max_steps_per_episode']:
            si=sid(s,w);a,_=choose(q,si,eval_action_rng,cfg,False,greedy=True)
            ns,r,done=eval_env.step(a);ret+=r;steps+=1;s=ns
            if done:success=1;break
        eval_rows.append({'maze':world_name,'bias_fraction':bias_fraction,'algorithm':algorithm,'seed':seed,'evaluation_episode':e,'steps':steps,'success':success,'extrinsic_return':ret})
    art=out/'artifacts'/world_name/f'bias_{bias_fraction:.2f}'/algorithm/f'seed_{seed:04d}';art.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(art/'q_table.npz',q_table=q,action_names=np.array(ACTION_NAMES))
    model_rows=[]
    for (si,a),outs in sorted(model.items()):
        for oi,(ni,r,d) in enumerate(outs):
            pred=corrupt_map.get((si,a),ni) if bias_fraction>0 and (si,a) in corrupted_pairs and not d else ni
            model_rows.append({'state':list(state(si,w)),'action':ACTION_NAMES[a],'outcome_index':oi,'observed_next_state':list(state(ni,w)),'planning_next_state':list(state(pred,w)),'reward':r,'done':d,'visit_count':visits[(si,a)],'corrupted_prediction':bool(bias_fraction>0 and (si,a) in corrupted_pairs and not d)})
    (art/'learned_model.json').write_text(json.dumps(model_rows,indent=2),encoding='utf-8')
    meta={'maze':world_name,'maze_rows':world['rows'],'shortest_path_length_moves':world['shortest'],'seed':seed,'bias_fraction':bias_fraction,'algorithm':algorithm,'config':cfg,'eligible_nonterminal_pairs':len(order),'reliability_counts':{f'{si}:{a}':{'matches':m,'observations':n,'smoothed_reliability':(m+1)/(n+2)} for (si,a),(m,n) in reliability_counts.items()} if reliability_aware else None,'predefined_corrupted_pairs':len(corrupted_pairs),'observed_corrupted_pairs':len(corrupted_pairs & seen),'realized_corruption_fraction_among_observed_eligible_pairs':len(corrupted_pairs & seen)/max(1,len(set(order_pair[:2] for order_pair in order)&seen)),'rng_design':'separate agent, training environment, evaluation environment, and greedy action streams','bias_mask_design':'nested prefixes of a per-maze/per-seed shuffled eligible state-action list; independent of algorithm visitation'}
    (art/'run_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return episode_rows,eval_rows

def write_csv(path,rows):
    if rows:
        with path.open('w',newline='',encoding='utf-8') as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
def mean_sd(vals):
    vals=[float(v) for v in vals if v is not None]
    return (statistics.mean(vals),statistics.stdev(vals) if len(vals)>1 else 0.) if vals else (None,None)

def main():
    p=argparse.ArgumentParser(description='Reliability-aware Dyna-Q experiment across multiple mazes and model-bias levels')
    p.add_argument('--seeds',type=int,default=10);p.add_argument('--episodes',type=int,default=200);p.add_argument('--max-steps',type=int,default=150);p.add_argument('--planning-steps',type=int,default=5);p.add_argument('--eval-episodes',type=int,default=10);p.add_argument('--output',default='results');p.add_argument('--bias-levels',default='0,0.05,0.10,0.20,0.30')
    args=p.parse_args()
    if min(args.seeds,args.episodes,args.max_steps,args.eval_episodes)<1 or args.planning_steps<0:p.error('counts must be positive (planning-steps may be zero)')
    levels=[float(x.strip()) for x in args.bias_levels.split(',')]
    if not levels or any(x<0 or x>0.8 for x in levels):p.error('bias levels must be between 0 and 0.8')
    if len(set(levels))!=len(levels):p.error('bias levels must be unique')
    levels=sorted(levels)
    cfg={'episodes':args.episodes,'max_steps_per_episode':args.max_steps,'evaluation_episodes':args.eval_episodes,'alpha':0.1,'gamma':0.95,'epsilon_fixed':0.2,'epsilon_min':0.05,'epsilon_max':0.8,'lambda_dispersion':5.0,'planning_steps':args.planning_steps,'step_penalty':-0.01,'goal_reward':1.0,'adaptive_epsilon_formula':'eps_min + (eps_max-eps_min)*exp(-lambda_dispersion*(max_a Q(s,a)-min_a Q(s,a)))','model_bias_scope':'planning updates only; real updates use observed transitions','reliability_method':'Agreement between learned-model prediction made before observing the outcome and subsequent observed next state; no prediction is scored for unseen pairs; estimator does not query true transition function or corruption labels','mask_design':'nested corruption masks across levels, predefined per maze and seed independently of algorithm visitation','rng_design':'separate agent, training environment, evaluation environment, and greedy action streams','note':'Adaptive epsilon is a heuristic based on action-value dispersion, not a calibrated uncertainty estimate. Reliability-aware Dyna uses pre-outcome predictions from its learned model and later observed outcomes to scale planning updates; it does not query the true transition function or corruption labels. This is an experimental heuristic, not a validated uncertainty estimator.'}
    algs=['q_learning','dyna_q_fixed','dyna_q_adaptive','dyna_q_reliability']
    run_id=datetime.now().strftime('reliability_aware_%Y%m%d_%H%M%S');out=Path(args.output)/run_id;out.mkdir(parents=True,exist_ok=False)
    record={'created_utc':datetime.now(timezone.utc).isoformat(),'run_id':run_id,'cli_args':vars(args),'config':cfg,'bias_levels':levels,'algorithms':algs,'mazes':{k:{'rows':v['rows'],'start':v['start'],'goal':v['goal'],'shortest_path_length_moves':v['shortest']} for k,v in WORLD.items()},'seed_values':list(range(args.seeds)),'planned_runs':len(WORLD)*len(levels)*len(algs)*args.seeds,'interpretation_guardrails':['Bias is applied to simulated planning next states only; real experience remains accurate.','Corruption levels use nested predefined masks within each maze and seed.','Q-learning is included at each nominal bias level as a repeated control; its updates do not use the planning model.','Reliability-aware planning is evaluated against fixed Dyna-Q, adaptive-epsilon Dyna-Q, and model-free Q-learning. Multiple mazes improve coverage but do not establish generality beyond these environments.']}
    (out/'config.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    eps=[];evs=[];t0=time.time();total=record['planned_runs'];done_runs=0
    for maze_name,world in WORLD.items():
      for level in levels:
       for alg in algs:
        print(f'[{done_runs+1}/{total}] maze={maze_name} bias={level:.2f} algorithm={alg}',flush=True)
        for seed in range(args.seeds):
         er,vr=run_one(maze_name,world,seed,level,alg,cfg,out);eps.extend(er);evs.extend(vr);done_runs+=1
    write_csv(out/'per_episode_results.csv',eps);write_csv(out/'greedy_evaluation_results.csv',evs)
    tail_start=max(1,int(args.episodes*.8));seed_rows=[]
    for maze_name in WORLD:
     for level in levels:
      for alg in algs:
       for seed in range(args.seeds):
        rr=[x for x in eps if x['maze']==maze_name and x['bias_fraction']==level and x['algorithm']==alg and x['seed']==seed and x['episode']>=tail_start]
        vv=[x for x in evs if x['maze']==maze_name and x['bias_fraction']==level and x['algorithm']==alg and x['seed']==seed]
        seed_rows.append({'maze':maze_name,'bias_fraction':level,'algorithm':alg,'seed':seed,'tail_steps_mean':statistics.mean(x['steps'] for x in rr),'tail_success_rate':statistics.mean(x['success'] for x in rr),'tail_extrinsic_return_mean':statistics.mean(x['extrinsic_return'] for x in rr),'tail_mean_epsilon':statistics.mean(x['mean_epsilon'] for x in rr if x['mean_epsilon'] is not None),'eval_success_rate':statistics.mean(x['success'] for x in vv),'eval_successful_steps_mean':statistics.mean(x['steps'] for x in vv if x['success']) if any(x['success'] for x in vv) else None})
    write_csv(out/'seed_level_summary.csv',seed_rows)
    summary=[]
    for maze_name in WORLD:
     for level in levels:
      for alg in algs:
       rr=[x for x in seed_rows if x['maze']==maze_name and x['bias_fraction']==level and x['algorithm']==alg]
       row={'maze':maze_name,'bias_fraction':level,'algorithm':alg,'n_seeds':args.seeds}
       for metric in ['tail_steps_mean','tail_success_rate','tail_extrinsic_return_mean','tail_mean_epsilon','eval_success_rate','eval_successful_steps_mean']:
        m,sd=mean_sd([x[metric] for x in rr]);row[metric+'_mean']=m;row[metric+'_sd_across_seeds']=sd
       summary.append(row)
    write_csv(out/'summary_by_maze.csv',summary)
    pooled=[]
    for level in levels:
     for alg in algs:
      rr=[x for x in seed_rows if x['bias_fraction']==level and x['algorithm']==alg]
      row={'bias_fraction':level,'algorithm':alg,'n_maze_seed_units':len(rr)}
      for metric in ['tail_steps_mean','tail_success_rate','tail_extrinsic_return_mean','eval_success_rate']:
       m,sd=mean_sd([x[metric] for x in rr]);row[metric+'_mean']=m;row[metric+'_sd_across_maze_seed_units']=sd
      pooled.append(row)
    write_csv(out/'pooled_summary.csv',pooled)
    # Plots pooled across maze-seed units, with each maze-seed as one unit.
    for metric,ylabel,fn in [('eval_success_rate','Greedy evaluation success','reliability_eval_success.png'),('tail_steps_mean','Final-window training steps','reliability_tail_steps.png'),('tail_success_rate','Final-window training success','reliability_train_success.png')]:
     fig,ax=plt.subplots(figsize=(8.5,5.5))
     for alg in algs:
      means=[];sds=[]
      for level in levels:
       vals=[x[metric] for x in seed_rows if x['bias_fraction']==level and x['algorithm']==alg]
       m,sd=mean_sd(vals);means.append(m);sds.append(sd)
      ax.errorbar([100*x for x in levels],means,yerr=sds,marker='o',capsize=3,label=alg)
     ax.set_xlabel('Predefined model-bias level (%)');ax.set_ylabel(ylabel);ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/fn,dpi=180);plt.close(fig)
    (out/'run_complete.json').write_text(json.dumps({'status':'completed','elapsed_seconds':round(time.time()-t0,2),'completed_runs':done_runs,'planned_runs':total,'files':['config.json','per_episode_results.csv','greedy_evaluation_results.csv','seed_level_summary.csv','summary_by_maze.csv','pooled_summary.csv','reliability_eval_success.png','reliability_tail_steps.png','reliability_train_success.png','artifacts/<maze>/bias_<level>/<algorithm>/seed_<seed>/{q_table.npz,learned_model.json,run_metadata.json}']},indent=2),encoding='utf-8')
    print('\nCompleted.');print('Output:',out.resolve());print('Pooled summary:',(out/'pooled_summary.csv').resolve())
if __name__=='__main__':main()
