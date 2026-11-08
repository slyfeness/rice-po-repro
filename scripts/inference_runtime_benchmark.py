"""Explicit one-GPU IEEE scheduler screen over a supplied saved-native manifest."""
import argparse,gc,json,os,time
from pathlib import Path

def main():
    from ricepo.artifacts import atomic_json
    from ricepo.audit import attempt
    from ricepo.vllm_two_stage import VLLMEngine,VLLMTwoStage
    from transformers import AutoTokenizer
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--seqs',type=int,required=True)
    p.add_argument('--batched',type=int,required=True);p.add_argument('--label',required=True)
    p.add_argument('--long',action='store_true')
    p.add_argument('--manifest',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    out=Path(a.output)/a.label
    out.mkdir(parents=True)
    manifest=json.loads(Path(a.manifest).read_text())
    model=a.model
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    policy='fixed-initial-ieee-runtime-screen-v1'
    options=dict(max_num_seqs=a.seqs,max_num_batched_tokens=a.batched,gpu_memory_utilization=.8,
                 enforce_eager=False,enable_prefix_caching=True,enable_chunked_prefill=True,attention_backend='TRITON_ATTN')
    with attempt(out):
        t=time.monotonic();engine=VLLMEngine(model,tok,policy,runtime_options=options);init=time.monotonic()-t
        metrics=[];generate=engine.llm.generate
        def capture(*args,**kwargs):
            values=generate(*args,**kwargs)
            for x in values:
                m=getattr(x,'metrics',None)
                metrics.append({'request_id':str(x.request_id),'metrics':None if m is None else
                    {k:getattr(m,k,None) for k in ('arrival_time','first_scheduled_time','first_token_time','finished_time','time_in_queue')}})
            return values
        engine.llm.generate=capture
        warm=0.;rows=manifest['requests'];results=[];phase_times={}
        for stage in ('think','summary'):
            r=next(x for x in rows if x['stage']==stage)
            t=time.monotonic();engine.sample([dict(prefix=tuple(r['prefix_token_ids']),budget=r['remaining_token_budget'],seed=r['seed'])],stage,policy)
            warm+=time.monotonic()-t
        metrics.clear();start=time.monotonic()
        for stage in ('think','summary'):
            batch=[r for r in rows if r['stage']==stage]
            req=[dict(prefix=tuple(r['prefix_token_ids']),budget=r['remaining_token_budget'],seed=r['seed']) for r in batch]
            t=time.monotonic();values=engine.sample(req,stage,policy);phase_times[stage]=time.monotonic()-t
            for row,value in zip(batch,values,strict=True):
                record={**row,**value};atomic_json(out/'responses'/(row['request_id']+'.json'),record);results.append(record)
        wall=time.monotonic()-start;tokens=sum(len(x['token_ids']) for x in results)
        result=dict(candidate=a.candidate,runtime=options,engine_initialization_seconds=init,warmup_seconds=warm,
            formal_wall_seconds=wall,generated_tokens=tokens,tokens_per_second=tokens/wall,completed_requests=len(results),
            stage_wall_seconds=phase_times,request_metrics=metrics,
            actual_active_concurrency_status='available only if engine returns request timing; configured concurrency is not occupancy',
            lengths=[len(x['token_ids']) for x in results])
        atomic_json(out/'RESULT.json',result)
        if a.long:
            thinks=sorted((r for r in rows if r['stage']=='think'),key=lambda r:r['request_id'])
            selected=thinks[:7]+[max(thinks,key=lambda r:len(r['prefix_token_ids']))]
            assert len({r['request_id'] for r in selected})==8
            atomic_json(out/'LONG_REQUEST_SELECTION.json',[{'request_id':r['request_id'],'seed':r['seed'],'prefix_tokens':len(r['prefix_token_ids'])} for r in selected])
            sampler=VLLMTwoStage(engine,tok,lambda:policy,4096,32768)
            t=time.monotonic();long=sampler.generate_many([tuple(r['prefix_token_ids']) for r in selected],[r['seed'] for r in selected]);seconds=time.monotonic()-t
            atomic_json(out/'LONG_NATIVE_RESULTS.json',long)
            atomic_json(out/'LONG_RESULT.json',dict(requests=8,wall_seconds=seconds,
                generated_tokens=sum(sum(x.action_mask) for x in long),
                result_types=[type(x).__name__ for x in long],shared_action_budget=4096,policy_version=policy))
        print(json.dumps({k:v for k,v in result.items() if k not in ('request_metrics','lengths')}),flush=True)
        engine.llm.llm_engine.engine_core.shutdown()

if __name__=='__main__':main()
