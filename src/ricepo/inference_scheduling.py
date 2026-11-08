"""Opt-in inference experiments, outside Trainer and optimizer execution."""
import time


def partition_requests(rows,workers):
    """Sticky deterministic grouping, balanced by frozen input+budget cost."""
    if type(workers) is not int or workers<1:raise ValueError('positive worker count required')
    if len({r['request_id'] for r in rows})!=len(rows):raise ValueError('duplicate request identities')
    groups={}
    for r in rows:groups.setdefault(r['group_key'],[]).append(r)
    bins=[[] for _ in range(workers)];loads=[0]*workers
    for key,group in sorted(groups.items(),key=lambda item:(-sum(len(r['prefix_token_ids'])+
            min(512,r['remaining_token_budget']) for r in item[1]),item[0])):
        target=min(range(workers),key=lambda i:(loads[i],i))
        bins[target].extend(r['request_id'] for r in group)
        loads[target]+=sum(len(r['prefix_token_ids'])+min(512,r['remaining_token_budget']) for r in group)
    return bins


def parallel_conditions(workers,partitions,output,*,retrieval=None,reset_cache=True,cap=512):
    """One coordinator clock; JNI retrieval remains on the caller's thread."""
    if len(workers)!=len(partitions) or any(not p for p in partitions):raise ValueError('nonempty aligned workers required')
    ids=sum(partitions,[])
    if len(ids)!=len(set(ids)):raise ValueError('duplicate scheduled request')
    if len({w.ready['policy'] for w in workers})!=1:raise ValueError('incompatible immutable worker weights')
    started=time.monotonic()
    for i,(w,part) in enumerate(zip(workers,partitions,strict=True)):
        w.start_conditions(part,output/f'worker-{i}',reset_cache=reset_cache,cap=cap)
    retrieved=retrieval() if retrieval is not None else None
    results=[w.finish_conditions() for w in workers];wall=time.monotonic()-started
    if sum(r['summary']['requests'] for r in results)!=len(ids):raise ValueError('incomplete result pool')
    tokens=sum(r['summary']['generated_tokens'] for r in results)
    return dict(common_wall_seconds=wall,generated_tokens=tokens,tokens_per_second=tokens/wall,
        requests=len(ids),request_ids=ids,partitions=partitions,workers=results,retrieval=retrieved,
        optimizer_steps=0,clock_basis='single coordinator monotonic clock; includes host dispatch, evidence I/O and retrieval')
