"""Existing fixed Retriever, initialized without a Trainer or optimizer."""
from pathlib import Path
from .state import read_json


def load_retrieval(deployment,recipe,binding,*,device,cache_root=None):
    from transformers import AutoTokenizer
    from .hf_runtime import _manifest,_domain_retriever
    from .retrieval import DiverEncoder,PyTrecReward
    from .formats import PromptRenderer
    d=dict(deployment);d['retriever_device']=device
    if cache_root is not None:d['cache_root']=str(cache_root)
    domains=set(d['domains'])
    _,examples,qrels=_manifest(d['train_manifest_path'],domains,expected_sha256=binding['manifest_sha256']['train'])
    tok=AutoTokenizer.from_pretrained(d['actor_model_path'],local_files_only=True,trust_remote_code=False)
    renderer=PromptRenderer(tok,Path(d['prompt_path']).read_text(),observation_top_k=recipe['observation_top_k'],
        observation_token_limit=recipe['observation_token_limit'],prefill_think=True)
    encoder=DiverEncoder.from_local(d)
    config=read_json(Path(d['output_root'])/'config.json')
    backends={}
    for domain,spec in d['domains'].items():
        backends[domain],_=_domain_retriever(domain,spec,d,recipe,config['model'],encoder,renderer,examples,qrels)
    return backends,PyTrecReward(qrels),encoder,renderer,examples


def retrieval_workload(backends,reward,requests,*,repetitions=4):
    import time,json
    result=[];start=time.monotonic()
    for repeat in range(repetitions):
        for r in requests:
            domain=json.loads(r['query_id'])[0]
            tick=time.monotonic()
            found=backends[domain].retrieve(r['query'],excluded_ids=frozenset(r['excluded_ids']))
            ids=[d.document_id for d in found.documents];score=reward.score(r['query_id'],ids)
            result.append(dict(request_id=r['request_id'],repeat=repeat,start=tick,end=time.monotonic(),
                seconds=time.monotonic()-tick,ids=ids,ndcg=score,
                saved_rank_equal=ids==r['expected_ids'],saved_ndcg_equal=abs(score-r['expected_ndcg'])<1e-12))
    return dict(wall_seconds=time.monotonic()-start,requests=len(result),results=result,
                all_saved_ranks_equal=all(x['saved_rank_equal'] for x in result),
                all_saved_ndcg_equal=all(x['saved_ndcg_equal'] for x in result))
