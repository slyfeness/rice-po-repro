import json,os,time
from pathlib import Path
from ricepo.continuation import scheduled_queries
from ricepo.hf_runtime import _manifest
from ricepo.artifacts import atomic_json,file_sha256
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('explicit Slurm allocation required')
R=Path(os.environ['RICE_PROJECT_ROOT']);out=R/'reports'/f'gemm-training-plan-{os.environ["SLURM_JOB_ID"]}';out.mkdir()
dep=json.loads((R/'runs/gemm-online-B-175853/deployment.json').read_text())
old=json.loads((R/'reports/gemm-independent-plan-175841/PRIVATE_VALIDATION_PLAN.json').read_text())
_,rows,_=_manifest(dep['train_manifest_path'],set(dep['domains']),expected_sha256=old['train_manifest_sha256'])
seed=20261107
plan=dict(kind='tf32_gemm_training_validation_v1',base_commit='92addec697190eff6e00ded7bfab8d01f5a6ae8c',frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),training_seed=seed,train_manifest_sha256=file_sha256(Path(dep['train_manifest_path'])),query_batches=[scheduled_queries(rows,seed,i,4) for i in range(3)],conditions=old['conditions'],risk=dict(first_outside_max=.01,later_outside_max=.02,extreme_max=.0005),speed_decision=dict(minimum_wall_reduction=.10,additional_requirement='higher measured engine generation token/s; report unequal outputs; no equal-work causality claim'),budget=dict(A_updates=1,B_updates=3,queries_per_update=4),old_qualification_plan_sha256=file_sha256(R/'reports/gemm-independent-plan-175841/PRIVATE_VALIDATION_PLAN.json'),selection='scheduled_queries existing stable seed rule, no outcome filtering')
atomic_json(out/'PRIVATE_PLAN.json',plan)
print('FROZEN',out.name,file_sha256(out/'PRIVATE_PLAN.json'),flush=True)
