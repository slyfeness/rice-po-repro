"""Recover one paid, uncommitted global update from verified P0 evidence.

No generation, tokenization, retrieval or model scoring occurs here. A new run
retains its real source/config binding and records the old source separately.
Only global update zero with a verified empty-Adam Initial state is supported.
"""
from dataclasses import fields,is_dataclass,asdict
import ast,json,types,typing,os,shutil
from pathlib import Path
from . import contracts as c
from .artifacts import atomic_json,file_sha256,stage_receipt,require_cluster_storage
from .audit import fingerprint,event,operation
from .state import read_json
from .failure_zero import validate_trajectory,validate_collections
from .rollout import BranchRollouts
from .credit import trajectory_advantages,build_advantages,select_anchors


def decode(tp,value):
    if value is None:
        if tp is type(None) or type(None) in typing.get_args(tp):return None
        raise ValueError('unexpected absent native field')
    origin=typing.get_origin(tp);args=typing.get_args(tp)
    if origin in (typing.Union,types.UnionType):
        for alternative in args:
            try:return decode(alternative,value)
            except (TypeError,ValueError,KeyError):pass
        raise ValueError('native field has no supported interpretation')
    if origin is tuple:
        if len(args)!=2 or args[1] is not Ellipsis:raise ValueError('unsupported native tuple schema')
        return tuple(decode(args[0],x) for x in value)
    if tp in (c.Generation,c.InvalidGeneration):
        result=c.generation_from_record(value)
        if not isinstance(result,tp):raise ValueError('native generation kind differs')
        return result
    if isinstance(tp,type) and is_dataclass(tp):
        names={f.name for f in fields(tp)}
        if set(value)!=names:raise ValueError('native record fields differ from supported schema')
        hints=typing.get_type_hints(tp)
        return tp(**{k:decode(hints[k],v) for k,v in value.items()})
    if tp is float and type(value) in (int,float):return value
    if tp in (int,str,bool) and type(value) is tp:return value
    raise ValueError('unsupported native value type')


def decode_trajectory(value,depth):
    result=decode(c.Trajectory,value);validate_trajectory(result,depth);return result


def reject_committed_source(root):
    root=Path(root)
    if (root/'completed.json').exists() or list(root.glob('iteration-*/optimizer_step_committed.receipt.json')):
        raise ValueError('cannot reuse a source with a committed optimizer step')


def reselect_anchors(grouped,entropies,budget):
    selected=[]
    for group in grouped.values():
        scoped={t.trajectory_id:entropies[t.trajectory_id] for t in group}
        selected.extend(select_anchors(group,scoped,top_k=sum(len(t.steps) for t in group))[:budget])
    return tuple(selected)


class CompletedCreditRecovery:
    def __init__(self,spec,binding,recipe,expected_queries):
        from .scale_training import verify_initial_state
        self.root=require_cluster_storage(spec['run']);self.directory=self.root/'iteration-000000'
        self.recipe=recipe;self.expected_queries=set(expected_queries);reject_committed_source(self.root)
        if sorted(p.name for p in self.root.glob('iteration-[0-9]*'))!=['iteration-000000']:
            raise ValueError('only the first uncommitted update is recoverable')
        old=read_json(self.root/'binding.json');self.old=old
        for key in ('dataset_receipt_sha256','manifest_sha256','model_sha256','retriever_sha256','prompt_sha256','behavior','pilot_frozen_schedule_sha256'):
            if old[key]!=binding[key]:raise ValueError('completed credit source differs: '+key)
        if read_json(self.root/'recipe.json')['values']!=recipe or fingerprint(read_json(self.root/'config.json'))!=old['config_sha256']:
            raise ValueError('source recipe or config differs')
        old_code=require_cluster_storage(spec['source'])/'src/ricepo'
        files={p.name:file_sha256(p) for p in sorted(old_code.glob('*.py'))}
        if files!=read_json(self.root/'sources.json') or fingerprint(files)!=old['source_sha256']:
            raise ValueError('old execution source identity differs')
        current=Path(__file__).parent
        for name in ('contracts.py','credit.py','objective.py','rollout.py','failure_zero.py','formats.py','retrieval.py','backends.py'):
            if file_sha256(current/name)!=files[name]:raise ValueError('recovery changed scientific code: '+name)
        def native_function(root):
            return next(n for n in ast.parse((root/'modeling.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_native_scores')
        if ast.dump(native_function(old_code),include_attributes=False)!=ast.dump(native_function(current),include_attributes=False):
            raise ValueError('recovery changed native scoring math')
        self.initial=verify_initial_state(self.root,fingerprint(old))
        # A missing commit receipt alone is not proof that step never ran.
        # The frozen old training code emits this durable event before step.
        for path in self.root.glob('attempts/*/events/*.json'):
            with path.open() as f:head=f.read(8192)
            if '"optimizer_minibatch_step"' in head:
                raise ValueError('ambiguous optimizer execution: step-start evidence exists')
        failures=list(self.root.glob('attempts/*/failure.json'))
        controlled=self.root/'controlled-credit-stop.json'
        if failures:
            if not any(read_json(p).get('type') in ('OutOfMemoryError','RuntimeError') and 'out of memory' in read_json(p).get('reason','').lower() for p in failures):
                raise ValueError('unsupported failure; no automatic credit replay')
        elif not (controlled.exists() and read_json(controlled).get('stopped_before_probability_risk') is True
                  and not (self.directory/'pilot-probability-risk.json').exists()):
            raise ValueError('missing confirmed pre-optimizer termination evidence')
        self.evidence={}
        self.old_policy=self.initial['policy_version']
        for stage,name in [('rollout_complete','trajectories.json'),('credit_complete','advantages.json')]:
            receipt=read_json(self.directory/(stage+'.receipt.json'))
            if receipt['policy_version']!=self.old_policy or receipt['artifacts'][name]!=file_sha256(self.directory/name):
                raise ValueError('completed stage identity changed: '+stage)
            self.evidence[name]=receipt['artifacts'][name]
        self.originals=tuple(decode_trajectory(t,recipe['rollout_depth']) for t in read_json(self.directory/'trajectories.json'))
        groups={}
        for t in self.originals:
            if t.policy_version!=self.old_policy:raise ValueError('mixed original policy')
            # Requires real saved q, including failed native actions.
            if c.trajectory_behavior_log_probs(t) is None:raise ValueError('saved original has no actual behavior probabilities')
            groups.setdefault(t.query_id,[]).append(t)
        if len({t.trajectory_id for t in self.originals})!=len(self.originals) or set(groups)!=self.expected_queries or any(len(v)!=recipe['trajectories_per_query'] for v in groups.values()):
            raise ValueError('incomplete original sampling group')
        self.collections=();self.anchors=()
        if recipe['method']=='ricepo':
            for stage,name in [('branches_complete','counterfactual-trajectories.json'),('entropy','anchors.json'),('entropy','summary-entropies.json')]:
                record=read_json(self.directory/(stage+'.receipt.json'))
                if record['policy_version']!=self.old_policy or record['artifacts'][name]!=file_sha256(self.directory/name):raise ValueError('cached entropy or branches changed')
                self.evidence[name]=record['artifacts'][name]
            self.anchors=tuple(decode(c.Anchor,a) for a in read_json(self.directory/'anchors.json'))
            self.collections=tuple(decode(BranchRollouts,x) for x in read_json(self.directory/'counterfactual-trajectories.json'))
            original_map={t.trajectory_id:t for t in self.originals}
            validate_collections(self.collections,self.anchors,original_map,recipe['total_branches_including_original'],recipe['rollout_depth'])
            entropies={t.trajectory_id:[float('nan')]*len(t.token_ids) for t in self.originals}
            for item in read_json(self.directory/'summary-entropies.json'):
                span=item['span'];entropies[item['trajectory_id']][span['start']:span['end']]=item['values']
            selected=reselect_anchors(groups,entropies,recipe['anchors_per_query'])
            if selected!=self.anchors:raise ValueError('saved Top-k anchors differ from saved full-vocabulary Summary entropy')
        elif recipe['method']!='grpo':raise ValueError('unsupported credit recovery method')
        comparable=[x.group for x in self.collections if all(o.local_reward is not None for o in x.group.outcomes)]
        parts=[]
        for query,group in groups.items():
            parts.append(build_advantages(group,[b for b in comparable if b.anchor.query_id==query],
                expected_branch_count=recipe['total_branches_including_original'],epsilon=recipe['normalization_epsilon'],
                local_normalization=recipe['local_advantage_normalization'],gate_mode=recipe['gate_mode'],
                variance_threshold=recipe['influence_threshold'],residual_threshold=recipe['residual_threshold'])
                if recipe['method']=='ricepo' else trajectory_advantages(group,epsilon=recipe['normalization_epsilon']))
        rows={key:(a,m) for part in parts for key,a,m in zip(part.trajectory_ids,part.advantages,part.generated_masks,strict=True)}
        self.batch=c.AdvantageBatch(tuple(t.trajectory_id for t in self.originals),tuple(rows[t.trajectory_id][0] for t in self.originals),
            tuple(rows[t.trajectory_id][1] for t in self.originals),tuple(r for part in parts for r in part.records))
        if fingerprint(asdict(self.batch))!=fingerprint(read_json(self.directory/'advantages.json')):raise ValueError('recomputed credit or masks differ from saved advantages')
        self.provenance=dict(source_alias=self.root.name,source_binding_sha256=fingerprint(old),source_code_sha256=old['source_sha256'],
            initial_state_complete_sha256=file_sha256(self.root/'initial-state/complete.json'),evidence_sha256=self.evidence,
            global_update=0,originals=len(self.originals),new_branches=sum(len(x.trajectories) for x in self.collections),
            new_sampling=False,independent_new_samples=False,source_optimizer_steps_proven_absent=True,
            policy='same uncommitted global update; native IDs/q/rewards/masks and CPU-rebuilt credit, not cross-update reuse')

    def restore_initial(self,optimizer,policy):
        import torch
        if policy!=self.old_policy:raise ValueError('actual restored P0 policy differs from paid sampling')
        state=torch.load(self.root/'initial-state/optimizer-and-rng.pt',map_location='cpu',weights_only=True)
        if (state['optimizer']['state'] or state['query_schedule_position']!=0 or state['global_update_index']!=-1
                or state['binding']!=fingerprint(self.old) or state['policy_version']!=policy or state['iteration']!=-1
                or state['seed']!=self.recipe['training_seeds'][0]
                or state['query_schedule_sha256']!=self.old['pilot_frozen_schedule_sha256']):
            raise ValueError('recovery Initial is not a verified untrained state')
        optimizer.load_state_dict(state['optimizer'])
        torch.random.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state_all(state['cuda_rng_all'])
        if not torch.equal(torch.random.get_rng_state(),state['cpu_rng']) or any(not torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),state['cuda_rng_all'],strict=True)):
            raise ValueError('Initial RNG restoration differs')

    def run(self,components,*,iteration,policy_version,output_root,recipe,branch_sink):
        if iteration!=0 or policy_version!=self.old_policy or recipe!=self.recipe:raise ValueError('credit recovery used at a different update/policy/recipe')
        reject_committed_source(self.root)
        directory=Path(output_root)/'iteration-000000';directory.mkdir(exist_ok=False)
        # New receipts keep original immutable files' real SHA and policy.
        names=['query-batch.json','trajectories.json','advantages.json','credit-diagnostics.json']
        stages=['rollout_complete','branches_complete','credit_complete']
        if recipe['method']=='ricepo':names+=['anchors.json','summary-entropies.json','counterfactual-trajectories.json'];stages+=['entropy']
        for name in names:
            source=self.directory/name
            if not source.exists():continue
            if name in self.evidence and file_sha256(source)!=self.evidence[name]:raise ValueError('replay source changed after validation')
            os.link(source,directory/name)
        for name in stages:
            receipt=read_json(self.directory/(name+'.receipt.json'))
            stage_receipt(directory,name,policy_version=policy_version,artifacts=receipt['artifacts'])
        atomic_json(directory/'completed-credit-recovery.json',self.provenance)
        branch_sink.extend(t for collection in self.collections for t in collection.trajectories)
        components.phase_barrier('credit','update')
        with operation('update',iteration=iteration,saved_native_credit_recovery=True):
            result=components.update(self.originals,self.batch,policy_version,directory)
        digest=atomic_json(directory/'update.json',result)
        stage_receipt(directory,'optimizer_step_committed',policy_version=result['new_policy_version'],artifacts={'update.json':digest})
        return result
