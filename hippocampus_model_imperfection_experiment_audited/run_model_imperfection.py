#!/usr/bin/env python3
"""Controlled model-imperfection experiment for hippocampal-inspired planning.

Conditions:
  accurate_deterministic: deterministic environment, no model impairment
  stochastic_transitions: intended action is replaced by a random action with probability p
  incomplete_model: a fixed fraction of observed state-action pairs are unavailable to planning
  biased_model: a fixed fraction of state-action model entries predict an incorrect next state

Algorithms:
  q_learning_fixed (model-free reference)
  dyna_q_fixed (fixed epsilon)
  dyna_q_adaptive (Q-dispersion heuristic epsilon)

The model impairment affects simulated planning updates only. Real interaction updates
always use the transition actually observed from the environment. All settings and
artifacts are saved to a timestamped folder. This is a research diagnostic, not proof
of a neuroscience mechanism.
"""
from __future__ import annotations
import argparse, csv, json, math, random, statistics, time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ACTIONS = {0:(-1,0), 1:(0,1), 2:(1,0), 3:(0,-1)}
ACTION_NAMES = ['up','right','down','left']
MAZE = ['S...#...', '.##.#.#.', '...#....', '.#...##.', '.###.#..', '...#.#..', '##...#..', '.......G']
ROWS, COLS = len(MAZE), len(MAZE[0])
START = next((r,c) for r,row in enumerate(MAZE) for c,ch in enumerate(row) if ch=='S')
GOAL = next((r,c) for r,row in enumerate(MAZE) for c,ch in enumerate(row) if ch=='G')
WALLS = {(r,c) for r,row in enumerate(MAZE) for c,ch in enumerate(row) if ch=='#'}
VALID = [(r,c) for r in range(ROWS) for c in range(COLS) if (r,c) not in WALLS]
N_STATES = ROWS*COLS

def sid(s): return s[0]*COLS+s[1]
def state(i): return (i//COLS, i%COLS)
def shortest_path():
    q=deque([(START,0)]); seen={START}
    while q:
        s,d=q.popleft()
        if s==GOAL:return d
        for dr,dc in ACTIONS.values():
            n=(s[0]+dr,s[1]+dc)
            if 0<=n[0]<ROWS and 0<=n[1]<COLS and n not in WALLS and n not in seen:
                seen.add(n);q.append((n,d+1))
    raise RuntimeError('Maze has no path to goal')

def deterministic_next(s, action):
    dr,dc=ACTIONS[int(action)]; n=(s[0]+dr,s[1]+dc)
    if not(0<=n[0]<ROWS and 0<=n[1]<COLS) or n in WALLS: return s
    return n

def make_impairment_masks(seed, cfg):
    """Predefine impairment by state-action pair, independently of algorithm visitation."""
    eligible=[]
    for s in VALID:
        if s == GOAL: continue
        for a in ACTIONS:
            ns=deterministic_next(s,a)
            eligible.append((sid(s),a,ns))
    incomplete_rng=random.Random(seed*1000003 + 41041)
    incomplete_n=round(len(eligible)*cfg['incomplete_fraction'])
    incomplete_pairs=set((si,a) for si,a,_ in incomplete_rng.sample(eligible, min(incomplete_n,len(eligible))))
    bias_candidates=[(si,a,ns) for si,a,ns in eligible if ns != GOAL]
    bias_rng=random.Random(seed*1000003 + 91009)
    biased_n=round(len(bias_candidates)*cfg['biased_fraction'])
    chosen=bias_rng.sample(bias_candidates, min(biased_n,len(bias_candidates)))
    corrupt_map={}
    non_goal=[sid(s) for s in VALID if s != GOAL]
    for si,a,ns in chosen:
        candidates=[x for x in non_goal if x != sid(ns)]
        if candidates:
            corrupt_map[(si,a)]=bias_rng.choice(candidates)
    return incomplete_pairs, corrupt_map, len(eligible), len(bias_candidates)

class GridWorld:
    def __init__(self, condition, cfg, rng):
        self.condition,self.cfg,self.rng=condition,cfg,rng; self.s=START
    def reset(self): self.s=START; return self.s
    def step(self, action):
        intended=int(action)
        if self.condition=='stochastic_transitions' and self.rng.random()<self.cfg['slip_probability']:
            action=self.rng.randrange(4)
        dr,dc=ACTIONS[int(action)]; n=(self.s[0]+dr,self.s[1]+dc)
        if not(0<=n[0]<ROWS and 0<=n[1]<COLS) or n in WALLS: n=self.s
        self.s=n; done=n==GOAL
        return n, (self.cfg['goal_reward'] if done else self.cfg['step_penalty']), done

def eps_for(qrow,cfg,adaptive):
    if not adaptive:return cfg['epsilon_fixed']
    dispersion=float(np.max(qrow)-np.min(qrow))
    return float(np.clip(cfg['epsilon_min']+(cfg['epsilon_max']-cfg['epsilon_min'])*math.exp(-cfg['lambda_dispersion']*dispersion),cfg['epsilon_min'],cfg['epsilon_max']))

def choose(q, s, rng, cfg, adaptive, greedy=False):
    eps=0.0 if greedy else eps_for(q[s],cfg,adaptive)
    if rng.random()<eps:return rng.randrange(4),eps
    best=np.flatnonzero(np.isclose(q[s],np.max(q[s])))
    return int(rng.choice(best.tolist())),eps

def update(q,s,a,r,ns,done,cfg):
    target=r if done else r+cfg['gamma']*float(np.max(q[ns]))
    q[s,a]+=cfg['alpha']*(target-q[s,a])

def run_one(seed, algorithm, condition, cfg, outdir):
    # Keep agent decisions/planning separate from environment stochasticity. This
    # prevents extra planning RNG draws from changing the environment's random stream.
    agent_rng=random.Random(seed + 120001)
    env_seed=seed + 220003 + sum(ord(c) for c in condition)
    env=GridWorld(condition,cfg,random.Random(env_seed))
    eval_env=GridWorld(condition,cfg,random.Random(seed + 330007 + sum(ord(c) for c in condition)))
    eval_rng=np.random.default_rng(seed + 440009)
    q=np.zeros((N_STATES,4),dtype=np.float64); visits=np.zeros((N_STATES,4),dtype=np.int64)
    planning=algorithm!='q_learning_fixed'; adaptive=algorithm=='dyna_q_adaptive'
    model={}; experienced=[]; seen=set(); episode_rows=[]
    unavailable,corrupt_map,eligible_pair_count,bias_eligible_count=make_impairment_masks(seed,cfg)
    for ep in range(1,cfg['episodes']+1):
        s=env.reset(); steps=0; ret=0.; success=0; epss=[]; plan_count=0; plan_attempted=0; plan_skipped=0
        while steps<cfg['max_steps_per_episode']:
            si=sid(s); a,eps=choose(q,si,agent_rng,cfg,adaptive); epss.append(eps)
            ns,r,done=env.step(a); ni=sid(ns); visits[si,a]+=1
            update(q,si,a,r,ni,done,cfg) # real transition is always correct
            pair=(si,a)
            if pair not in seen:
                seen.add(pair); experienced.append(pair)
            if pair not in model: model[pair]=[]
            model[pair].append((ni,float(r),bool(done)))
            if planning and experienced:
                for _ in range(cfg['planning_steps']):
                    plan_attempted += 1
                    ps,pa=agent_rng.choice(experienced)
                    if condition=='incomplete_model' and (ps,pa) in unavailable:
                        plan_skipped += 1
                        continue
                    # Sample from observed outcomes; this represents stochastic transitions
                    # with an empirical outcome distribution rather than only the last sample.
                    pni,pr,pdone=agent_rng.choice(model[(ps,pa)])
                    if condition=='biased_model' and (ps,pa) in corrupt_map and not pdone:
                        pni=corrupt_map[(ps,pa)]
                        # Corrupt nonterminal next-state predictions only; terminal targets
                        # ignore the next state, so corrupting them would be ineffectual.
                    update(q,ps,pa,pr,pni,pdone,cfg); plan_count+=1
            ret+=r;steps+=1;s=ns
            if done:success=1;break
        episode_rows.append({'condition':condition,'algorithm':algorithm,'seed':seed,'episode':ep,'steps':steps,'success':success,'extrinsic_return':ret,'mean_epsilon':float(np.mean(epss)) if epss else None,'planning_updates_attempted':plan_attempted,'planning_updates_applied':plan_count,'planning_updates_skipped_model_unavailable':plan_skipped,'observed_state_action_pairs':len(seen),'impaired_pairs_observed':len(unavailable.intersection(seen)) if condition=='incomplete_model' else len(corrupt_map.keys() & seen) if condition=='biased_model' else 0})
    eval_rows=[]
    for e in range(1,cfg['evaluation_episodes']+1):
        s=eval_env.reset(); steps=0;ret=0.;success=0;visited=set()
        while steps<cfg['max_steps_per_episode']:
            si=sid(s); a,_=choose(q,si,eval_rng,cfg,False,greedy=True)
            ns,r,done=eval_env.step(a); ret+=r;steps+=1;s=ns
            if done:success=1;break
            # no artificial loop termination; step cap is explicit
            visited.add(s)
        eval_rows.append({'condition':condition,'algorithm':algorithm,'seed':seed,'evaluation_episode':e,'steps':steps,'success':success,'extrinsic_return':ret})
    artifact=Path(outdir)/'artifacts'/condition/algorithm/f'seed_{seed:04d}';artifact.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(artifact/'q_table.npz',q_table=q,action_names=np.array(ACTION_NAMES))
    model_rows=[]
    for (si,a),outcomes in sorted(model.items()):
        for outcome_index,(ni,r,d) in enumerate(outcomes):
            model_rows.append({'state':list(state(si)),'action':ACTION_NAMES[a],'outcome_index':outcome_index,'observed_next_state':list(state(ni)),'planning_next_state':list(state(corrupt_map.get((si,a),ni) if condition=='biased_model' and (si,a) in corrupt_map and not d else ni)),'reward':r,'done':d,'real_visit_count':int(visits[si,a]),'masked_from_planning':condition=='incomplete_model' and (si,a) in unavailable,'corrupted_prediction':condition=='biased_model' and (si,a) in corrupt_map and not d})
    (artifact/'learned_model.json').write_text(json.dumps(model_rows,indent=2),encoding='utf-8')
    (artifact/'run_metadata.json').write_text(json.dumps({'seed':seed,'condition':condition,'algorithm':algorithm,'config':cfg,'maze':MAZE,'shortest_path_length_moves':shortest_path(),'model_pairs_observed':len(model),'incomplete_pairs_predefined':len(unavailable) if condition=='incomplete_model' else 0,'corrupted_pairs_predefined':len(corrupt_map) if condition=='biased_model' else 0,'eligible_state_action_pairs':eligible_pair_count,'bias_eligible_pairs':bias_eligible_count,'incomplete_pairs_observed':len(unavailable.intersection(seen)) if condition=='incomplete_model' else 0,'corrupted_pairs_observed':len(corrupt_map.keys() & seen) if condition=='biased_model' else 0,'rng_design':'separate agent, training environment, and evaluation environment streams'},indent=2),encoding='utf-8')
    return episode_rows,eval_rows

def write_csv(path,rows):
    if rows:
        with open(path,'w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mean_sd(xs):
    xs=[float(x) for x in xs]
    return (statistics.mean(xs),statistics.stdev(xs) if len(xs)>1 else 0.) if xs else (None,None)

def main():
    p=argparse.ArgumentParser(description='Controlled Dyna model-imperfection experiment')
    p.add_argument('--seeds',type=int,default=10);p.add_argument('--episodes',type=int,default=200)
    p.add_argument('--max-steps',type=int,default=150);p.add_argument('--planning-steps',type=int,default=5)
    p.add_argument('--eval-episodes',type=int,default=10);p.add_argument('--output',default='results')
    p.add_argument('--slip-probability',type=float,default=.15);p.add_argument('--incomplete-fraction',type=float,default=.25)
    p.add_argument('--biased-fraction',type=float,default=.20)
    args=p.parse_args()
    if args.seeds<1 or args.episodes<1 or args.max_steps<1 or args.eval_episodes<1 or args.planning_steps<0:p.error('counts must be positive (planning-steps may be zero)')
    for name,val in [('slip-probability',args.slip_probability),('incomplete-fraction',args.incomplete_fraction),('biased-fraction',args.biased_fraction)]:
        if not 0<=val<=1:p.error(f'--{name} must be in [0,1]')
    cfg={'episodes':args.episodes,'max_steps_per_episode':args.max_steps,'evaluation_episodes':args.eval_episodes,'alpha':.1,'gamma':.95,'epsilon_fixed':.2,'epsilon_min':.05,'epsilon_max':.8,'lambda_dispersion':5.,'planning_steps':args.planning_steps,'step_penalty':-.01,'goal_reward':1.,'slip_probability':args.slip_probability,'incomplete_fraction':args.incomplete_fraction,'biased_fraction':args.biased_fraction,'adaptive_epsilon_formula':'eps_min + (eps_max-eps_min)*exp(-lambda_dispersion*(max_a Q(s,a)-min_a Q(s,a)))','model_imperfection_scope':'planning updates only; real updates use observed transitions', 'mask_design':'impairment masks precomputed per seed over valid state-action pairs, independently of algorithm visitation', 'rng_design':'agent, training environment, and evaluation environment use separate reproducible random streams','note':'Adaptive epsilon is a heuristic based on action-value dispersion, not a calibrated uncertainty estimate.'}
    conditions=['accurate_deterministic','stochastic_transitions','incomplete_model','biased_model']
    algorithms=['q_learning_fixed','dyna_q_fixed','dyna_q_adaptive']
    run_id=datetime.now().strftime('model_imperfection_%Y%m%d_%H%M%S');out=Path(args.output)/run_id;out.mkdir(parents=True,exist_ok=False)
    record={'created_utc':datetime.now(timezone.utc).isoformat(),'run_id':run_id,'cli_args':vars(args),'config':cfg,'conditions':conditions,'algorithms':algorithms,'maze':MAZE,'start':START,'goal':GOAL,'shortest_path_length_moves':shortest_path(),'seed_values':list(range(args.seeds)),'interpretation_guardrails':['The stochastic condition is a stochastic environment, not a corrupted internal model.','Incomplete model omits selected observed state-action pairs from planning; it does not fabricate a transition.','Biased model corrupts nonterminal predicted next states for a preselected subset of pairs during planning only; terminal next states are not corrupted because they do not affect terminal Q targets.','Do not infer general superiority from a single maze; use multiple environments in later validation.']}
    (out/'config.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    all_ep=[];all_ev=[];t0=time.time()
    for cond in conditions:
        for alg in algorithms:
            print(f'Running condition={cond} algorithm={alg}',flush=True)
            for seed in range(args.seeds):
                ep,ev=run_one(seed,alg,cond,cfg,out);all_ep.extend(ep);all_ev.extend(ev)
    write_csv(out/'per_episode_results.csv',all_ep);write_csv(out/'greedy_evaluation_results.csv',all_ev)
    # Seed-level summaries are the independent units; avoid treating episodes as independent replicates.
    seed_rows=[]
    tail_start=max(1,int(args.episodes*.8))
    for cond in conditions:
        for alg in algorithms:
            for seed in range(args.seeds):
                rows=[x for x in all_ep if x['condition']==cond and x['algorithm']==alg and x['seed']==seed and x['episode']>=tail_start]
                ev=[x for x in all_ev if x['condition']==cond and x['algorithm']==alg and x['seed']==seed]
                seed_rows.append({'condition':cond,'algorithm':alg,'seed':seed,'tail_steps_mean':statistics.mean(x['steps'] for x in rows),'tail_success_rate':statistics.mean(x['success'] for x in rows),'tail_extrinsic_return_mean':statistics.mean(x['extrinsic_return'] for x in rows),'tail_mean_epsilon':statistics.mean(x['mean_epsilon'] for x in rows if x['mean_epsilon'] is not None),'eval_success_rate':statistics.mean(x['success'] for x in ev),'eval_successful_steps_mean':statistics.mean(x['steps'] for x in ev if x['success']) if any(x['success'] for x in ev) else None})
    write_csv(out/'seed_level_summary.csv',seed_rows)
    summary=[]
    for cond in conditions:
        for alg in algorithms:
            rows=[x for x in seed_rows if x['condition']==cond and x['algorithm']==alg]
            item={'condition':cond,'algorithm':alg,'n_seeds':args.seeds}
            for metric in ['tail_steps_mean','tail_success_rate','tail_extrinsic_return_mean','tail_mean_epsilon','eval_success_rate','eval_successful_steps_mean']:
                vals=[x[metric] for x in rows if x[metric] is not None];m,sd=mean_sd(vals);item[metric+'_mean']=m;item[metric+'_sd_across_seeds']=sd
            summary.append(item)
    write_csv(out/'summary_results.csv',summary)
    for metric,ylabel,filename in [('success','Training success rate','learning_curve_success.png'),('steps','Steps per episode','learning_curve_steps.png'),('extrinsic_return','Extrinsic return','learning_curve_return.png')]:
        fig,axs=plt.subplots(2,2,figsize=(12,8));axs=axs.ravel()
        for ax,cond in zip(axs,conditions):
            for alg in algorithms:
                rr=[x for x in all_ep if x['condition']==cond and x['algorithm']==alg]
                xs=[];means=[];lows=[];highs=[]
                for epn in range(1,args.episodes+1):
                    vals=[x[metric] for x in rr if x['episode']==epn]
                    if vals:
                        m,sd=mean_sd(vals);xs.append(epn);means.append(m);lows.append(m-sd);highs.append(m+sd)
                ax.plot(xs,means,label=alg);ax.fill_between(xs,lows,highs,alpha=.12)
            ax.set_title(cond);ax.set_xlabel('Episode');ax.set_ylabel(ylabel);ax.grid(alpha=.2)
        axs[0].legend(fontsize=7);fig.suptitle(f'{ylabel} by environment/model condition (mean ± SD across seeds)');fig.tight_layout();fig.savefig(out/filename,dpi=180);plt.close(fig)
    (out/'run_complete.json').write_text(json.dumps({'status':'completed','elapsed_seconds':round(time.time()-t0,2),'files':['config.json','per_episode_results.csv','greedy_evaluation_results.csv','seed_level_summary.csv','summary_results.csv','learning_curve_success.png','learning_curve_steps.png','learning_curve_return.png','artifacts/<condition>/<algorithm>/seed_<seed>/{q_table.npz,learned_model.json,run_metadata.json}']},indent=2),encoding='utf-8')
    print('\nCompleted.');print('Shortest path:',shortest_path(),'moves');print('Results:',out.resolve());print('Summary:',(out/'summary_results.csv').resolve())
if __name__=='__main__': main()
