import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ricepo.scale_training import validate_contract, risk_limit, cadence, preserve_unpersisted_tail
from ricepo.asset_identity import VerifiedModelIdentity


def recipe(stage='training', method='ricepo'):
    return dict(method=method,queries_per_update=32 if stage=='capacity' else 128,
        total_updates=1 if stage=='capacity' else 10,training_seeds=[20261108],trajectories_per_query=4,
        anchors_per_query=4,total_branches_including_original=6,rollout_depth=5,
        local_advantage_normalization='center_only',influence_threshold=.05,residual_threshold=.03,
        actor_learning_rate=1e-6,actor_kl_coefficient=.001,clip_epsilon=.2,ppo_epochs=1,microbatch_size=1,
        optimizer_minibatch_size='full_batch',gradient_accumulation='full_batch',loss_reduction='sequence_mean',
        query_composition={'mode':'original_and_summary','separator':'\n'},temperature=1.,top_p=1.,top_k=0,
        max_context_tokens=32768,max_new_tokens_per_step=4096,checkpoint_every_updates=2,development_every_updates=4,
        checkpoint_selection='best_development_macro',eval_samples_per_query=4,
        native_span_parser='two_stage_env_summary_v1',format_failure_policy='invalid_action_terminal_zero_v1')


class ScaleContracts(unittest.TestCase):
    def test_recover_before_first_checkpoint_from_verified_initial(self):
        from ricepo.scale_training import publish_initial_state,verify_initial_state,recomputation_generations
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);publish_initial_state(root,dict(optimizer={},policy_version='p0',binding='b'))
            (root/'iteration-000000').mkdir();(root/'iteration-000000/optimizer_step_committed.receipt.json').write_text('{}')
            result=preserve_unpersisted_tail(root,'b')
            self.assertEqual(result['persisted_updates'],0);self.assertEqual(result['discarded_computed_updates'],[1])
            self.assertEqual(recomputation_generations(root),{0:1})
            self.assertEqual(verify_initial_state(root,'b')['iteration'],-1)
            (root/'initial-state/optimizer-and-rng.pt').write_bytes(b'tampered')
            with self.assertRaises(ValueError):verify_initial_state(root,'b')
    def test_recovery_unpublished_checkpoint_stage_is_validated_as_existing_runtime(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cp=root/'iteration-000000/checkpoint';cp.mkdir(parents=True);(cp/'complete.json').write_text('{}')
            with patch('ricepo.scale_training.verify_checkpoint',return_value=dict(iteration=0,policy_version='p1')) as check:
                self.assertEqual(preserve_unpersisted_tail(root,'b')['persisted_updates'],1)
                self.assertTrue(check.call_args.kwargs['allow_missing_stage'])
    def test_explicit_recomputation_namespace_keeps_seed_but_not_duplicate_request_key(self):
        from ricepo.audit import operation,logical_request_id,stable_seed
        seed=stable_seed(1,'original',2,'domain','query',0,1)
        with operation('scope',iteration=2):normal=logical_request_id('p','q','2-t',1,seed)
        with operation('scope',iteration=2,recovery_generation=1):recovery=logical_request_id('p','q','2-t',1,seed)
        self.assertNotEqual(normal,recovery);self.assertEqual(seed,stable_seed(1,'original',2,'domain','query',0,1))
    def test_lifecycle_rejects_stale_worker_and_exits_before_HF_residency(self):
        from types import SimpleNamespace
        from ricepo.vllm_lifecycle import PhaseSeparatedGenerator
        g=object.__new__(PhaseSeparatedGenerator);g.version=lambda:'new';g.worker=SimpleNamespace(ready={'policy':'old'})
        with self.assertRaisesRegex(RuntimeError,'stale'):g.prepare_generation()
        events=[];g.worker=SimpleNamespace(close=lambda:events.append('closed'))
        g.actor=SimpleNamespace(to=lambda d:events.append('actor'));g.reference=SimpleNamespace(to=lambda d:events.append('reference'))
        g.device='assigned';g.moves=[];g.prepare_scoring();self.assertEqual(events,['closed','actor','reference'])
    def test_new_scale_contract_keeps_old_diagnostic_separate(self):
        frozen=dict(kind='biology_scale_training_v1',training_seed=20261108,data_counts=dict(train=128,development=32,test=103))
        for stage in ('capacity','training','initial'):
            for method in ('ricepo','grpo'):validate_contract(stage,frozen,recipe(stage,method))
        wrong=recipe();wrong['queries_per_update']=4
        with self.assertRaises(ValueError):validate_contract('training',frozen,wrong)
        for key,value in [('rollout_depth',1),('total_updates',11),('influence_threshold',.04),('query_composition',{'mode':'summary_only'}),('loss_reduction','token_mean')]:
            wrong=recipe();wrong[key]=value
            with self.assertRaises(ValueError):validate_contract('training',frozen,wrong)
    def test_risk_cadence_tenth_update_and_single_global_batch(self):
        self.assertEqual([risk_limit(i) for i in (0,1,9)],[.01,.02,.02])
        for bad in (-1,10,True):
            with self.assertRaises(ValueError):risk_limit(bad)
        self.assertEqual([i for i in range(1,11) if cadence(i,10)['checkpoint']],[2,4,6,8,10])
        self.assertEqual([i for i in range(1,11) if cadence(i,10)['development']],[4,8,10])
    def test_identity_reuse_rechecks_mutation_and_added_file(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'weight').write_bytes(b'one')
            identities=iter([{'id':1},{'id':2},{'id':3}])
            with patch('ricepo.asset_identity.local_model_identity',side_effect=lambda p:next(identities)) as read:
                cache=VerifiedModelIdentity();self.assertEqual(cache.get(root),{'id':1});self.assertEqual(cache.get(root),{'id':1})
                self.assertEqual(read.call_count,1)
                (root/'weight').write_bytes(b'longer');self.assertEqual(cache.get(root),{'id':2})
                (root/'template').write_text('new');self.assertEqual(cache.get(root),{'id':3});self.assertEqual(read.call_count,3)
    def test_recovery_preserves_tail_and_only_resumes_verified_persisted_step(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for i in range(3):(root/f'iteration-{i:06d}').mkdir()
            cp=root/'iteration-000001'/'checkpoint';cp.mkdir();(cp/'complete.json').write_text('{}')
            (root/'iteration-000002'/'optimizer_step_committed.receipt.json').write_text('{}')
            with patch('ricepo.scale_training.verify_checkpoint',return_value=dict(iteration=1,policy_version='p2')):
                result=preserve_unpersisted_tail(root,'binding')
            self.assertEqual(result['persisted_updates'],2);self.assertEqual(result['discarded_computed_updates'],[3])
            self.assertFalse((root/'iteration-000002').exists());self.assertTrue(list((root/'recovery-archives').rglob('optimizer_step_committed.receipt.json')))
            with patch('ricepo.scale_training.verify_checkpoint',return_value=dict(iteration=1,policy_version='p2')):
                self.assertEqual(preserve_unpersisted_tail(root,'binding')['discarded_computed_updates'],[])
    def test_initial_wins_equal_development_score_and_tamper_rejected(self):
        from ricepo.artifacts import file_sha256
        from ricepo.state import selected_so_far
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'binding.json').write_text(json.dumps(dict(manifest_sha256={'development':'a'*64})))
            def report(path,policy):
                path.write_text(json.dumps(dict(scope='complete',split='development',policy_version=policy,
                    manifest_sha256='a'*64,expected_total=32,evaluated_total=32,macro=.2)))
            initial=root/'initial-summary.json';report(initial,'p0')
            (root/'binding.json').write_text(json.dumps(dict(manifest_sha256={'development':'a'*64},
                common_initial=dict(report_sha256=file_sha256(initial),policy_version='p0',actor_path='initial'))))
            (root/'initial-selection.json').write_text(json.dumps(dict(kind='frozen_initial_source',iteration=-1,
                checkpoint='initial',policy_version='p0',development_macro=.2,report_path=str(initial),report_sha256=file_sha256(initial))))
            t=root/'iteration-000003';t.mkdir();report(t/'summary.json','p4')
            (t/'development_complete.receipt.json').write_text(json.dumps(dict(policy_version='p4',artifacts=dict(relative_report='summary.json',sha256=file_sha256(t/'summary.json')))))
            self.assertEqual(selected_so_far(root,'best_development_macro')['iteration'],-1)
            initial.write_text('{}')
            with self.assertRaises(ValueError):selected_so_far(root,'best_development_macro')
    def test_nested_timing_is_exclusive_and_does_not_double_count(self):
        from ricepo.audit import attempt,operation
        with tempfile.TemporaryDirectory() as d:
            with attempt(Path(d)),patch('ricepo.audit.perf_counter',side_effect=[0.,1.,3.,5.]):
                with operation('outer'):
                    with operation('inner'):pass
            rows=[json.loads(p.read_text()) for p in Path(d).rglob('events/*.json')]
            done={r['phase']:r for r in rows if r['kind']=='completed'}
            self.assertEqual(done['inner']['exclusive_elapsed_seconds'],2.)
            self.assertEqual(done['outer']['exclusive_elapsed_seconds'],3.)
            self.assertEqual(sum(r['exclusive_elapsed_seconds'] for r in done.values()),5.)


if __name__=='__main__':unittest.main()
