"""CPU arithmetic scenarios only; does not claim measured end-to-end gains."""
import argparse,json,statistics,csv,hashlib
from pathlib import Path
from scripts.gpu_probe_common import write
from scripts.gpu_probe_export import csvwrite,jobs

def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    dp=json.loads(a.dp.read_text());prec=json.loads(a.precision.read_text())
    drows=[r for record in dp['results'] if record.get('rank')==0 for r in record['rows']]
    prows=[r for record in prec['results'] if record.get('count')==64 for r in record['rows']]
    means={p:statistics.mean(r['HF_wall_seconds'] for r in prows if r['precision']==p) for p in ('ieee','tf32','bf16')}
    dp_ratios={}
    for count in (64,512):
        single=[r['HF_wall_seconds'] for r in drows if r['count']==count and r['kind']=='single']
        dual=[r['HF_wall_seconds'] for r in drows if r['count']==count and r['kind']=='dual']
        if single and dual:dp_ratios[str(count)]=statistics.mean(single)/statistics.mean(dual)
    ratio=dp_ratios.get('512',dp_ratios.get('64'))
    components=dict(HF=6881,generation=1107,retrieval=155,lifecycle=296,checkpoint=188,unclassified=315)
    # User's rounded decomposition sums to8942 rather than8949.
    remainder=8949-sum(components.values());fixed=8949-components['HF']-components['generation']
    rows=[]
    for p in ('ieee','tf32','bf16'):
        for parallel in (False,True):
            if parallel and ratio is None:continue
            hf_speed=(means['ieee']/means[p])*(ratio if parallel else 1)
            for dual_generation in (False,True):
                generation_speed=1.575 if dual_generation else 1.
                predicted=fixed+components['HF']/hf_speed+components['generation']/generation_speed
                rows.append(dict(method='GRPO',query_count=128,precision=p,dual_HF=parallel,dual_generation=dual_generation,HF_speed_factor=hf_speed,generation_speed_factor=generation_speed,estimated_seconds=predicted,estimated_minutes=predicted/60,evidence='ESTIMATED',joint_HF_configuration_measured=not parallel or p=='ieee',full512_precision_measured=p=='ieee'))
    historical_rows=list(csv.DictReader(a.historical_costs.open()))
    rice=[r for r in historical_rows if r['queries']=='32' and 'ricepo' in r['run']]
    if rice:
        stages={r['category']:float(r['exclusive_seconds']) for r in rice}
        total=float(rice[0]['run_wall_seconds']);hf=stages['HF_SCORING_PPO'];gen=stages['GENERATION_HOST_WRAPPER']
        for scenario in list(rows):
            estimate=total-hf-gen+hf/scenario['HF_speed_factor']+gen/scenario['generation_speed_factor']
            rows.append({**scenario,'method':'RICE_PO_historical32_proxy','query_count':32,'estimated_seconds':estimate,'estimated_minutes':estimate/60,'joint_HF_configuration_measured':False,'full512_precision_measured':False,'factor_source':'saved GRPO originals; RICE branch scoring not measured'})
    result=dict(all_end_to_end_scenarios_are_estimates=True,precision_ratio_workload='Sub64 means of two measured repetitions',DP_ratio_workload='512' if '512' in dp_ratios else '64',measured_HF_means_seconds=means,measured_DP_ratios=dp_ratios,historical_GRPO_rounded_seconds=components,reported_total_seconds=8949,rounding_or_omitted_remainder_seconds=remainder,scenarios=rows,historical_costs_sha256=hashlib.sha256(a.historical_costs.read_bytes()).hexdigest(),assumptions=['6881s historical HF is rescaled using ratios measured under A+C2; this is a conditional scenario, not a direct measurement of the old full-sequence implementation','DP and precision speed factors are multiplied only as an unmeasured combination assumption','historical dual-engine1.575x is reused; no dual-engine generation ran in this task','initialization, checkpoint and other fixed terms held constant; offline HF timing excludes optimizer.step and full parameter comparison','Sub64 precision ratios are not full512 precision measurements','RICE32 proxy uses an older32-query run including evaluation; its branch-only HF work has not been timed under these changes. It is not a128-query prediction or a same-budget GRPO comparison'],RICE_PO=dict(estimated_total_seconds=None,reason='No clean128-query RICE stage decomposition:11901s is a failed/recovery completion attempt, not production HF baseline. GRPO costs cannot identify extra branch generation, branch scoring and entropy costs.',symbolic_seconds='HF_original/s_HF + HF_branches/s_branch + generation/1.575 + retrieval + entropy_credit + lifecycle + checkpoint + other',missing=['clean original-HF and branch-HF exclusive seconds','clean RICE generation and entropy/credit exclusive seconds','new DP/precision behavior on counterfactual scoring is not measured']))
    write(a.output/'RESULT.json',result);write(a.output/'NUMERICAL.json',dict(type='CPU arithmetic, no new model numerics',measured_DP_ratios=dp_ratios,measured_precision_means=means))
    csvwrite(a.output/'TIMING.csv',rows);csvwrite(a.output/'GPU_UTIL.csv',[],fields=['not_applicable_CPU_only'])
    csvwrite(a.output/'SLURM_JOBS.csv',jobs(a.jobs),fields=['job_id','state','elapsed_seconds','allocated_gpus','GPU_hours','exit_code'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dp',type=Path,required=True);p.add_argument('--precision',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--jobs',nargs='+',required=True);p.add_argument('--historical-costs',type=Path,required=True);main(p.parse_args())
