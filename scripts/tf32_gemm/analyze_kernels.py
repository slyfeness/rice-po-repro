# Public driver: explicit Slurm execution only; no downloads.
import json,os,re,hashlib,collections
from pathlib import Path

if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm allocation required')
R=Path(os.environ['RICE_PROJECT_ROOT']);P=R/'benchmarks/tf32-gemm-probe-175836'
O=R/'reports'/f'tf32-gemm-kernels-{os.environ["SLURM_JOB_ID"]}';O.mkdir()
result={};index=[]
for label in ('A','B'):
    state=json.loads((P/label/'STATE.json').read_text());kernels=collections.defaultdict(lambda:[0,0.]);ops=collections.Counter()
    for p in (P/label/'profiler').glob('*.json'):
        raw=p.read_bytes();index.append(dict(alias=f'{label}/{p.name}',sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
        trace=json.loads(raw)
        for e in trace['traceEvents']:
            if e.get('cat')=='kernel':
                k=kernels[e['name']];k[0]+=1;k[1]+=e.get('dur',0.)
            if e.get('cat')=='cpu_op' and re.search('linear|addmm|::mm',e.get('name','')):ops[e['name']]+=1
    gemm=[dict(name=k,count=v[0],gpu_duration_us=v[1]) for k,v in sorted(kernels.items()) if re.search('gemm|gemv|matmul',k,re.I)]
    result[label]=dict(worker_state=[{k:v for k,v in s.items() if k!='pid'} for s in state['state']],
        initialization_seconds=state['initialization_seconds'],manifest_sha256=state['manifest_sha256'],
        model_payload_sha256=state['model_identity']['payload_sha256'],gemm_kernels=gemm,aten_operators=dict(ops),
        kernel_duration_note='tiny real-model probe only; summed kernel durations are not full workload makespan')
a={k['name'] for k in result['A']['gemm_kernels']};b={k['name'] for k in result['B']['gemm_kernels']}
result['new_B_gemm_kernels']=sorted(b-a);result['removed_A_gemm_kernels']=sorted(a-b)
result['control_effect']='MEASURED' if b-a and any(re.search('tf32|s1688',k,re.I) for k in b-a) else 'INCONCLUSIVE'
result['evidence_index']=index
(O/'KERNEL_COMPARISON.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
