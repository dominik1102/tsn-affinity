#!/usr/bin/env python
from __future__ import annotations
import argparse, numpy as np, os

from clbench.core.registry import TaskRegistry
from clbench.adapters.cartpole import CartPoleAdapter
from clbench.adapters.atari import AtariAdapter
from clbench.io.serialize import load_task_specs
from clbench.benchmark.runner import make_tasks, describe_tasks, BenchmarkResults
from clbench.benchmark.metrics import StandardCLMetrics
from clbench.benchmark.metrics_extra import per_step_report
from clbench.io.run_logger import build_run_dir, save_json, save_matrix_csv, bench_short, save_task_gen_json

TaskRegistry.register("cartpole", CartPoleAdapter())
TaskRegistry.register("atari", AtariAdapter())

def random_train(env, total_steps: int):
    steps, obs, info = 0, *env.reset()
    while steps<total_steps:
        a=env.action_space.sample(); obs,r,done,trunc,info=env.step(a); steps+=1
        if done or trunc: obs,info=env.reset()

def random_eval(env, episodes:int)->float:
    rets=[]
    for _ in range(episodes):
        obs,info=env.reset(); total=0.0
        while True:
            a=env.action_space.sample(); obs,r,done,trunc,info=env.step(a); total+=r
            if done or trunc: break
        rets.append(total)
    return float(np.mean(rets)) if rets else 0.0

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--spec",required=True)
    p.add_argument("--episodes-eval",type=int,default=5)
    p.add_argument("--steps-per-task",type=int,default=1000)
    p.add_argument("--runs-root",type=str,default="runs")
    p.add_argument("--tag",type=str,default="")
    args=p.parse_args()

    specs=load_task_specs(args.spec)
    is_atari=any((s.params or {}).get("game","").startswith("ALE/") for s in specs)
    bench="atari" if is_atari else "cartpole"
    envs=make_tasks(bench,specs)
    names=list(envs.keys()); n=len(names); P=np.zeros((n,n),dtype=np.float32)
    spec_tag=os.path.splitext(os.path.basename(args.spec))[0]
    run_dir=build_run_dir(args.runs_root,bench,None,tag=args.tag or spec_tag)
    for i,(name,env) in enumerate(envs.items()):
        random_train(env,args.steps_per_task)
        for j,(n2,env2) in enumerate(envs.items()):
            P[i,j]=random_eval(env2,args.episodes_eval)
        save_task_gen_json(run_dir,i+1,{"step":i+1,"task_name":name,"row_scores":[float(x) for x in P[i,:i+1]]})
    results=BenchmarkResults(name=f"{bench}:{args.spec}",task_names=names,perf_matrix=P)
    metrics=StandardCLMetrics.compute(results)
    save_json(os.path.join(run_dir,"results.json"),{"name":results.name,"task_names":names,"perf_matrix":P.tolist(),"metrics":metrics})
    save_matrix_csv(os.path.join(run_dir,"matrix.csv"),names,P)
    steps=per_step_report(names,P)
    save_json(os.path.join(run_dir,"per_step.json"),{"per_step":steps})
    with open(os.path.join(run_dir,"per_step.csv"),"w",encoding="utf-8",newline="") as f:
        import csv; w=csv.DictWriter(f,fieldnames=list(steps[0].keys())); w.writeheader(); w.writerows(steps)
    save_json(os.path.join(run_dir,f"rez_{bench_short(bench)}.json"),{"metrics":metrics})
    print(f"[saved] {run_dir}")

if __name__=="__main__":
    main()
