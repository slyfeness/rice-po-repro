"""Bounded offline S0/S1/S2/S3 sequence inside one two-GPU allocation.

No Slurm submission, model download, sampling, retrieval or automatic training.
The caller must supply a pre-frozen private plan and unused private output root.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def main(args):
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(devices)!=2 or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('explicit two-device Slurm allocation required')
    args.output.mkdir(parents=True,exist_ok=False)
    script=Path(__file__).with_name('hf_compute_benchmark.py')
    processes={}
    def launch(scheme,device,step=False):
        root=args.output/'winner-step' if step else args.output
        env={**os.environ,'CUDA_VISIBLE_DEVICES':devices[device]}
        log=(args.output/(scheme+('-step' if step else '')+'.private.log')).open('w')
        cmd=[sys.executable,'-u',str(script),'--plan',str(args.plan),'--output',str(root),'--scheme',scheme]
        if step:cmd.append('--step')
        p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        processes[(scheme,step)]=(p,log)
        return p
    def finish(scheme,step=False):
        p,log=processes[(scheme,step)];code=p.wait();log.close()
        path=(args.output/'winner-step' if step else args.output)/scheme/'RESULT.json'
        result=json.loads(path.read_text()) if path.is_file() else dict(status='PROCESS_FAILED',returncode=code,correctness_passed=False)
        print('CANDIDATE_TERMINAL',scheme,step,result['status'],flush=True)
        return result
    launch('S0',0);results={'S0':finish('S0')}
    if not results['S0'].get('correctness_passed'):
        raise RuntimeError('baseline did not complete; no candidates expanded')
    launch('S1',0);launch('S2',1)
    results.update(S1=finish('S1'),S2=finish('S2'))
    if all(results[s].get('correctness_passed') for s in ('S1','S2')):
        launch('S3',0);results['S3']=finish('S3')
    base=[r['HF_total_seconds'] for r in results['S0']['repeats']]
    gates=json.loads(args.plan.read_text())['speed_gate']
    eligible=[]
    for name,result in results.items():
        if not result.get('correctness_passed'):continue
        measured=[r['HF_total_seconds'] for r in result['repeats']]
        ratios=[a/b for a,b in zip(base,measured)]
        result['speedup_repeats']=ratios
        result['speedup_median']=statistics.median(base)/statistics.median(measured)
        result['speed_passed']=(result['speedup_median']>=gates['median_HF_total_speedup_min']
                                and min(ratios)>=gates['each_repeat_speedup_min'])
        if result['speed_passed']:eligible.append(name)
    winner=min(eligible,key=lambda s:statistics.median(r['HF_total_seconds'] for r in results[s]['repeats'])) if eligible else None
    decision=dict(results=results,winner=winner,scope='offline fixed native histories; no online RICE update')
    if winner:
        launch(winner,0,step=True);decision['winner_step']=finish(winner,step=True)
    (args.output/'DECISION_PRIVATE.json').write_text(json.dumps(decision,indent=2,allow_nan=False))
    print('OFFLINE_BENCHMARK_COMPLETE',winner,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
