"""Slurm-local phase sequencing, gates and bounded time-limit handoff only.

This script does not allocate GPUs itself. A planned durable-checkpoint stop
may submit one afterany successor using the caller's frozen private sbatch file.
Failures never trigger blind retries. No credentials or raw artifacts are sent.
"""
import argparse,json,os,signal,subprocess,sys
from pathlib import Path


def main(args):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm allocation required')
    from ricepo.artifacts import atomic_json,file_sha256
    from ricepo.scale_training import run_tag
    R=Path(os.environ['RICE_PROJECT_ROOT']);source=Path(os.environ['RICE_SOURCE_ROOT'])
    plan=Path(os.environ['RICE_SCALE_PLAN']);frozen=json.loads(plan.read_text())
    child=[None]
    def relay(*_):
        if child[0] is not None and child[0].poll() is None:child[0].send_signal(signal.SIGUSR1)
    signal.signal(signal.SIGUSR1,relay)
    def execute(script,arguments=(),env=None):
        child[0]=subprocess.Popen([sys.executable,'-u',str(source/'scripts/scale_training'/script),*arguments],env=env)
        code=child[0].wait();child[0]=None;return code
    def restore(stage):
        tag=run_tag(stage,args.method)
        env={**os.environ,'RICE_RESTORE_RUN':str(R/'runs'/tag),'RICE_RESTORE_ITERATION':'0' if stage=='capacity' else '9'}
        code=execute('restore.py',env=env)
        if code:raise RuntimeError('independent restore failed; phase remains blocked')
        record=R/'reports'/f'biology-scale-restore-{tag}-{os.environ["SLURM_JOB_ID"]}.json'
        result=json.loads(record.read_text())
        if result['status']!='PASS' or result['newly_executed_optimizer_steps']!=0:raise ValueError('invalid restore result')
        atomic_json(R/'runs'/tag/f'{stage}-restore-complete.json',dict(status='PASS',evidence=str(record),sha256=file_sha256(record)))
    if args.stage=='training':
        for method in ('ricepo','grpo'):
            cap=R/'runs'/f'biology-scale-v1-capacity-{method}'
            if json.loads((cap/'capacity-complete.json').read_text())['status']!='capacity_update_complete':raise ValueError('capacity gate not passed')
            receipt=json.loads((cap/'capacity-restore-complete.json').read_text())
            if receipt['status']!='PASS' or file_sha256(Path(receipt['evidence']))!=receipt['sha256']:raise ValueError('capacity restore changed')
        if not (Path(frozen['common_initial_root'])/'initial-evaluation-complete.json').is_file():raise ValueError('common Initial missing')
    arguments=[args.stage,args.method]+(['--resume'] if args.resume else [])
    code=execute('worker.py',arguments)
    if code==75 and args.stage=='training':
        batch=Path(os.environ['RICE_SCALE_SBATCH'])
        command=['sbatch','--parsable','--dependency=afterany:'+os.environ['SLURM_JOB_ID'],
                 '--job-name=rice-scale-'+args.method,'--time=48:00:00',str(batch),'training',args.method,'--resume']
        # A missing submission acknowledgement is ambiguous: never submit twice.
        submission=R/'reports'/f'biology-scale-handoff-{args.method}-{os.environ["SLURM_JOB_ID"]}.json'
        if submission.exists():raise ValueError('handoff already recorded')
        answer=subprocess.check_output(command,text=True).strip();job=answer.split(';')[0]
        if not job.isdigit():raise ValueError('ambiguous Slurm acknowledgement; inspect before retry')
        atomic_json(submission,dict(parent_job=os.environ['SLURM_JOB_ID'],successor_job=job,status='SUBMITTED',
            reason='measured-workload deadline or Slurm warning; resume verified durable state, no additional update budget'))
        print('PLANNED_HANDOFF_SUBMITTED',job,flush=True);return
    if code:raise RuntimeError('scale worker failed with code '+str(code))
    if args.stage=='capacity':
        restore('capacity')
        if args.method=='grpo':
            initial=Path(frozen['common_initial_root'])
            if not (initial/'initial-evaluation-complete.json').is_file():
                if execute('worker.py',['initial','ricepo']):raise RuntimeError('common Initial failed')
    elif args.stage=='training':restore('training')
    print('SCALE_PHASE_COMPLETE',args.stage,args.method,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['capacity','training']);p.add_argument('method',choices=['ricepo','grpo']);p.add_argument('--resume',action='store_true')
    main(p.parse_args())
