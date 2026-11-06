"""Bounded replay of one failed, uncommitted iteration from native action evidence."""
from dataclasses import asdict
from pathlib import Path

from .artifacts import file_sha256
from .audit import fingerprint,event
from .contracts import generation_from_record,InvalidGeneration
from .state import read_json


class VerifiedActionRecovery:
    def __init__(self,sources,binding,plan):
        if not isinstance(sources,list) or len(sources)!=1:
            raise ValueError('recovery v1 requires one explicit failed source')
        spec=sources[0];root=Path(spec['run']);code=Path(spec['source'])/'src/ricepo'
        if (root/'completed.json').exists() or list(root.glob('iteration-*/optimizer_step_committed.receipt.json')):
            raise ValueError('cannot replay a source with a committed optimizer update')
        previous=read_json(root/'binding.json')
        for key in ('config_sha256','recipe_fingerprint','dataset_receipt_sha256','manifest_sha256',
                    'model_sha256','retriever_sha256','prompt_sha256','behavior','continuation'):
            if previous.get(key)!=binding.get(key):raise ValueError('recovery source differs: '+key)
        if (fingerprint(read_json(root/'config.json'))!=previous['config_sha256']
                or read_json(root/'recipe.json')['values']!=plan):
            raise ValueError('recovery source config or recipe differs')
        files={p.name:file_sha256(p) for p in sorted(code.glob('*.py'))}
        if files!=read_json(root/'sources.json') or fingerprint(files)!=previous['source_sha256']:
            raise ValueError('recovery source code identity differs')
        failures=list(root.glob('attempts/*/failure.json'))
        if len(failures)!=1:raise ValueError('recovery requires one identified original failure')
        failure=read_json(failures[0])
        if failure.get('reason')!='unmapped native byte token':
            raise ValueError('unsupported partial-action recovery cause')
        failed=failure['operation'];self.failed_key=(failed['policy_version'],failed['seed'])
        self.actions={};self.paid={};self.stages={};self.expected={};self.anchors={}
        evidence={}
        def read(path):
            evidence[str(path.relative_to(root))]=file_sha256(path);return read_json(path)
        for p in root.glob('requests/*/first.json'):
            identity=read(p)['identity'];key=(identity['policy_version'],identity['seed'])
            if key in self.paid and self.paid[key]!=identity:raise ValueError('ambiguous paid request identity')
            self.paid[key]=identity
        for p in root.glob('attempts/*/events/*.json'):
            e=read_json(p)
            if e['kind'] not in ('two_stage_action_record','two_stage_stage_saved'):continue
            evidence[str(p.relative_to(root))]=file_sha256(p)
            key=(e['policy_version'],e['seed'])
            if e['kind']=='two_stage_action_record':
                if key not in self.paid or self.paid[key]['prefix_sha256']!=e['prefix_sha256']:
                    raise ValueError('saved action has no matching paid request')
                g=generation_from_record(e['generation'])
                if key in self.actions and self.actions[key]!=g:raise ValueError('conflicting saved actions')
                self.actions[key]=g
            else:
                stage_key=(*key,e['prefix_sha256'],e['stage_name'])
                if stage_key in self.stages and self.stages[stage_key]!=e:raise ValueError('conflicting saved stages')
                self.stages[stage_key]=e
        if set(self.paid)-set(self.actions)!={self.failed_key}:
            raise ValueError('missing paid actions beyond the identified partial failure')
        if any(version!=binding['continuation']['parent_policy_version'] for version,_ in self.paid):
            raise ValueError('paid actions do not all use the restored parent policy')
        for p in root.glob('iteration-*/trajectories.json'):
            for t in read(p):self.expected[t['trajectory_id']]=t
        for p in root.glob('attempts/*/items/counterfactual/*.json'):
            t=read(p);self.expected[t['trajectory_id']]=t
        for p in root.glob('iteration-*/anchors.json'):
            for a in read(p):self.anchors[a['trajectory_id'],a['depth']]=a
        self.provenance=dict(source_alias=root.name,source_binding_sha256=fingerprint(previous),
            source_code_sha256=previous['source_sha256'],evidence_sha256=fingerprint(evidence),
            complete_native_actions=len(self.actions),paid_requests=len(self.paid),partial_actions=1,
            original_partial_summary_ids_available=False,
            partial_policy='same prefix, seed and actual policy; saved Think IDs/RNG must match; original terminal cause must recur')

    def _key(self,prefix,seed,version):
        key=(version,seed)
        if key in self.paid and self.paid[key]['prefix_sha256']!=fingerprint(tuple(prefix)):
            raise ValueError('paid recovery request prefix changed')
        return key

    def __call__(self,prefix,seed,version):
        return self.actions.get(self._key(prefix,seed,version))

    def paid_reexecution(self,prefix,seed,version):
        key=self._key(prefix,seed,version)
        return key in self.paid and key not in self.actions

    def validate_stage(self,history,seed,version,stage,ids,lps,before,after):
        expected=self.stages.get((version,seed,fingerprint(tuple(history)),stage))
        if expected is None:return
        if (tuple(expected['native_token_ids'])!=tuple(ids)
                or expected['rng_before_sha256']!=before or expected['rng_after_sha256']!=after):
            raise ValueError('recovered native stage or RNG differs from saved evidence')
        from .probability_check import compare_log_probs
        check=compare_log_probs(expected['sampled_log_probs'],lps,[True]*len(ids))
        if not check['within_bf16_tolerance']:raise ValueError('recovered stage probability mismatch')
        event('recovery_saved_stage_verified',stage_name=stage,check=check)

    def validate_action(self,prefix,seed,version,generation):
        key=self._key(prefix,seed,version)
        if key==self.failed_key and (not isinstance(generation,InvalidGeneration)
                                    or generation.category!='native_token_unmapped'):
            raise ValueError('partial recovery terminal cause changed; no successful resampling allowed')
        if key==self.failed_key:event('recovery_partial_action_verified',generation=generation,
                                      source_binding_sha256=self.provenance['source_binding_sha256'])

    def validate_trace(self,trace):
        old=self.expected.get(trace.trajectory_id)
        if old is None:return
        for name in ('query_id','policy_version','final_reward','initial_ndcg','last_valid_ndcg','termination'):
            if getattr(trace,name)!=old.get(name):raise ValueError('recovered trajectory changed: '+name)
        if trace.token_ids!=tuple(old['token_ids']) or trace.generated_mask!=tuple(old['generated_mask']):
            raise ValueError('recovered native trajectory IDs or mask changed')
        if len(trace.steps)!=len(old['steps']):raise ValueError('recovered trajectory horizon changed')
        for step,saved in zip(trace.steps,old['steps'],strict=True):
            if (step.local_reward!=saved['local_reward'] or step.generation.summary_text!=saved['generation']['summary_text']
                    or [d.document_id for d in step.retrieval.documents]!=[d['document_id'] for d in saved['retrieval']['documents']]
                    or asdict(step.summary_span)!=saved['summary_span'] or asdict(step.reasoning_span)!=saved['reasoning_span']):
                raise ValueError('deterministic retrieval or native spans changed during recovery')
        event('recovery_native_trace_verified',trajectory_sha256=fingerprint(trace.trajectory_id))

    def check_anchor(self,anchor):
        if anchor.trajectory_id not in self.expected:return
        old=self.anchors.get((anchor.trajectory_id,anchor.depth))
        if old is None or any(asdict(anchor)[k]!=old[k] for k in ('query_id','reasoning_span','summary_span')):
            raise ValueError('recovery entropy anchor changed')
