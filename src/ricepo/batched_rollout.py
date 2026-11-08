"""Wavefront scheduling around the unchanged native rollout transition.

Generation and already-ready legal retrievals may be batched. Each action is consumed once by
the same validation, deterministic retrieval and failure-terminal machinery.
There is no decoding/re-encoding, resampling or history/cache shared by rows.
"""
from dataclasses import dataclass
from copy import copy

from .audit import operation, saved_item, fingerprint, check_scheduled_request
from .contracts import InvalidGeneration


@dataclass(frozen=True)
class OriginalRequest:
    query_id:str
    trajectory_id:str
    original_query:str
    initial_prefix_token_ids:tuple[int,...]
    seeds:tuple[int,...]
    old_policy_version:str
    initial_ndcg:float|None=None

    def serial_kwargs(self):
        return dict(query_id=self.query_id,trajectory_id=self.trajectory_id,original_query=self.original_query,
            initial_prefix_token_ids=self.initial_prefix_token_ids,seeds=self.seeds,
            old_policy_version=self.old_policy_version,initial_ndcg=self.initial_ndcg)


class _OneAction:
    def __init__(self,owner,prefix,seed,generation):
        self.owner,self.prefix,self.seed,self.generation=owner,prefix,seed,generation
        self.used=False
    @property
    def policy_version(self):return self.owner.policy_version
    def generate(self,prefix,*,seed):
        if self.used or tuple(prefix)!=self.prefix or seed!=self.seed:
            raise ValueError('batched action identity or consumption differs')
        self.used=True
        return self.generation


class _OneRetrieval:
    def __init__(self,request,result):
        self.request,self.result,self.used=request,result,False
    def retrieve(self,query,*,excluded_ids):
        if self.used or query!=self.request['query'] or excluded_ids!=self.request['excluded_ids']:
            raise ValueError('batched retrieval identity or consumption differs')
        self.used=True
        return self.result


def continue_many(runner,states):
    """States have the exact keyword contract of RolloutRunner._continue."""
    if not states or len({s['trajectory_id'] for s in states})!=len(states):
        raise ValueError('nonempty distinct batch request identities required')
    owner=runner.generator
    states=[dict(s) for s in states]
    results=[None]*len(states);active=list(range(len(states)))
    while active:
        prefixes=[tuple(states[i]['token_ids']) for i in active]
        seeds=[states[i]['seeds'][0] for i in active]
        for i in active:runner._check_policy(states[i]['policy_version'])
        identities=[dict(query_id=states[i]['query_id'],trajectory_id=states[i]['trajectory_id'],
                        depth=len(states[i]['previous_steps'])+1,seed=seed,
                        prefix_sha256=fingerprint(prefix),policy_version=states[i]['policy_version'])
                    for i,prefix,seed in zip(active,prefixes,seeds,strict=True)]
        for identity in identities:check_scheduled_request(identity)
        saved_item('batch_requests',identities,identities)
        kind='original_generation_batch' if states[active[0]]['honor_stop'] else 'branch_generation_batch'
        with operation(kind,requests=identities,input_tokens=sum(map(len,prefixes))):
            generations=(owner.generate_many_with_identities(prefixes,seeds,identities)
                         if hasattr(owner,'generate_many_with_identities') else owner.generate_many(prefixes,seeds))
        if len(generations)!=len(active):raise ValueError('batch omitted a scheduled action')
        retrievals={}
        if hasattr(runner.retriever,'retrieve_many'):
            ready=[]
            for i,g in zip(active,generations,strict=True):
                if isinstance(g,InvalidGeneration):continue
                runner._validate_generation(g)
                runner._check_length(tuple(states[i]['token_ids'])+tuple(g.token_ids))
                ready.append((i,dict(query=runner.query_composer.compose(states[i]['original_query'],g.summary_text),
                    excluded_ids=runner.excluded_ids_by_query.get(states[i]['query_id'],frozenset()))))
            if ready:
                with operation('retrieval_ready_wave',requests=len(ready)):
                    values=runner.retriever.retrieve_many([r for _,r in ready])
                if len(values)!=len(ready):raise ValueError('retrieval batch omitted a scheduled query')
                retrievals={i:_OneRetrieval(request,value) for (i,request),value in zip(ready,values,strict=True)}
        following=[]
        for i,prefix,seed,g in zip(active,prefixes,seeds,generations,strict=True):
            runner._check_policy(states[i]['policy_version'])
            one=copy(runner);one.generator=_OneAction(owner,prefix,seed,g)
            if i in retrievals:one.retriever=retrievals[i]
            state=states[i];depth=len(state['previous_steps'])+1
            t=one._continue(**{**state,'seeds':(seed,),'final_depth':depth})
            saved_item('batch_partial_trajectory',[t.trajectory_id,depth],t)
            results[i]=t
            if t.failure is None and t.termination!='caller_stop' and depth<state['final_depth']:
                state.update(token_ids=t.token_ids,generated_mask=t.generated_mask,
                             previous_steps=t.steps,seeds=state['seeds'][1:])
                following.append(i)
        active=following
    return tuple(results)


def run_many(runner,requests):
    states=[]
    for r in requests:
        prefix=tuple(r.initial_prefix_token_ids)
        if not prefix or any(type(x) is not int or x<0 for x in prefix):raise ValueError('invalid native prefix')
        if not r.query_id or not r.trajectory_id or not r.original_query:raise ValueError('missing request identity')
        runner._check_length(prefix)
        states.append(dict(query_id=r.query_id,trajectory_id=r.trajectory_id,original_query=r.original_query,
            policy_version=r.old_policy_version,token_ids=prefix,generated_mask=(False,)*len(prefix),
            previous_steps=(),seeds=runner._validate_seeds(r.seeds,runner.rules.depth),
            final_depth=runner.rules.depth,honor_stop=True,initial_ndcg=r.initial_ndcg))
    return continue_many(runner,states)
