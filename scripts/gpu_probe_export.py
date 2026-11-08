"""CPU-only aggregate of small private result files; no native histories exported."""
import argparse,csv,json,subprocess,hashlib
from pathlib import Path
from scripts.gpu_probe_common import write

def csvwrite(p,rows,fields=None):
    keys=fields or list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def jobs(ids):
    raw=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(ids),'--format=JobIDRaw,State,ElapsedRaw,AllocTRES%200,ExitCode'],text=True)
    out=[]
    for line in raw.splitlines():
        if not line.strip():continue
        job,state,seconds,tres,exitcode=line.split('|')[:5]
        allocated=dict(x.split('=',1) for x in tres.split(',') if '=' in x)
        gpu=int(allocated.get('gres/gpu','0'));elapsed=int(seconds)
        out.append(dict(job_id=job,state=state,elapsed_seconds=elapsed,allocated_gpus=gpu,GPU_hours=elapsed*gpu/3600,exit_code=exitcode))
    return out

def main(a):
    a.output.mkdir(parents=True,exist_ok=False);results=[];util=[];hashes={}
    for directory in a.inputs:
        for p in sorted(directory.glob('RESULT*.json')):
            results.append(json.loads(p.read_text()));hashes[p.name+'@'+str(len(hashes))]=hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.glob('GPU_UTIL*.jsonl'):
            util.extend(json.loads(line) for line in p.read_text().splitlines() if line)
    if a.only_count:
        for r in results:r['rows']=[row for row in r.get('rows',[]) if row.get('count',r.get('count'))==a.only_count]
        util=[row for row in util if 'Sub'+str(a.only_count)+'-' in row['label']]
    timing=[];numeric=[]
    for r in results:
        for row in r.get('rows',[]):
            timing.append(dict(rank=r.get('rank'),count=row.get('count',r.get('count')),kind=row.get('kind',row.get('precision')),repeat=row['repeat'],HF_wall_seconds=row['HF_wall_seconds'],allocated_peak_bytes=row['allocated_peak_bytes'],reserved_peak_bytes=row['reserved_peak_bytes'],optimizer_step_seconds=row.get('optimizer_step_seconds'),OOM=row['OOM']))
            numeric.append({k:v for k,v in row.items() if k not in ['GPU_util','rank_forward_backward_seconds','rank_gradient_sum_seconds']})
    pairs=[]
    for result in results:
        if result.get('rank')!=0:continue
        baseline={(r['count'],r['repeat']):r for r in result['rows'] if r['kind']=='single'}
        for row in result['rows']:
            key=(row['count'],row['repeat'])
            if row['kind']=='dual' and key in baseline:
                b=baseline[key];pairs.append(dict(count=row['count'],repeat=row['repeat'],single_seconds=b['HF_wall_seconds'],dual_seconds=row['HF_wall_seconds'],speedup=b['HF_wall_seconds']/row['HF_wall_seconds'],loss_difference=row['normalized_loss']-b['normalized_loss']))
    write(a.output/'RESULT.json',dict(paired_measurements=pairs,results=results,private_result_sha256=hashes,measurement_scope='saved native GRPO update2, actual P1 Actor and P0 Reference, A fused observation and C2; no training-resume or numerical acceptance thresholds',notes=['precision gradients are sums; DP gradients divide by global sequence count','first measured repetition includes any first-use effects; both repetitions reported','noise half differences reuse per-item IEEE gradients with CPUfloat64 summation; no inference-generation costs included']))
    csvwrite(a.output/'TIMING.csv',timing,fields=['rank','count','kind','repeat','HF_wall_seconds','allocated_peak_bytes','reserved_peak_bytes','optimizer_step_seconds','OOM'])
    csvwrite(a.output/'GPU_UTIL.csv',util,fields=['rank','label','seconds','utilization_percent','memory_MiB','power_W','error'])
    write(a.output/'NUMERICAL.json',dict(observations=numeric,half_splits=[dict(count=r.get('count'),noise=r['half_split_noise']) for r in results if 'half_split_noise' in r]))
    csvwrite(a.output/'SLURM_JOBS.csv',jobs(a.jobs),fields=['job_id','state','elapsed_seconds','allocated_gpus','GPU_hours','exit_code'])
    print(json.dumps(dict(results=len(results),timing_rows=len(timing),GPU_samples=len(util))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--jobs',nargs='+',required=True);p.add_argument('--only-count',type=int,default=0);main(p.parse_args())
