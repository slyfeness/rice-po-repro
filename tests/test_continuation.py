"""Continuation identity and global schedule; no model assets required."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from ricepo.audit import fingerprint
from ricepo.artifacts import atomic_json, file_sha256
from ricepo.continuation import parent_run
from ricepo.continuation import continuation_recipe, scheduled_queries, validate_state_position


class ContinuationTests(unittest.TestCase):
    def parent_fixture(self, directory):
        root=Path(directory)/'parent'; root.mkdir()
        source=Path(directory)/'source'; package=source/'src'/'ricepo'; package.mkdir(parents=True)
        (package/'original.py').write_text('frozen=True\n')
        sources={'original.py':file_sha256(package/'original.py')}
        plan=dict(total_updates=1,temperature=1,training_seeds=[42],queries_per_update=4)
        config={'execution':{'enabled':True}}
        deployment={'output_root':str(root),'prompt_path':'old','cache_root':'old','actor_precision':'float32'}
        binding=dict(config_sha256=fingerprint(config),recipe_fingerprint='recipe',
                     source_sha256=fingerprint(sources),deployment_sha256=fingerprint(deployment),
                     dataset_receipt_sha256='data',manifest_sha256={'train':'train'},
                     model_sha256='model',retriever_sha256='retriever',prompt_sha256='prompt',behavior={'precision':'float32'})
        for name,value in [('config',config),('recipe',{'values':plan,'fingerprint':'recipe'}),
                           ('sources',sources),('binding',binding),('deployment',deployment),
                           ('completed',{'completed_updates':1,'last_policy_version':'p','selected_checkpoint':{'development_macro':.2}})]:
            atomic_json(root/(name+'.json'),value)
        d=root/'iteration-000000';cp=d/'checkpoint';cp.mkdir(parents=True)
        atomic_json(cp/'complete.json',{'iteration':0,'policy_version':'p'})
        atomic_json(d/'iteration_complete.receipt.json',{})
        atomic_json(d/'update.json',{'optimizer_steps':1})
        atomic_json(d/'optimizer_step_committed.receipt.json',{'policy_version':'p','artifacts':{'update.json':file_sha256(d/'update.json')}})
        child={**deployment,'output_root':str(Path(directory)/'child'),'parent_run_root':str(root),
               'parent_source_root':str(source),'continuation_schedule_path':str(Path(directory)/'schedule.json')}
        train=['q'+str(i) for i in range(128)]
        atomic_json(child['continuation_schedule_path'],dict(global_start=1,root_seed=42,
            train_manifest_sha256='train',batches=[scheduled_queries(train,42,i,4) for i in (1,2)]))
        return child,binding,{**plan,'total_updates':2},train

    def test_parent_is_read_only_and_source_transition_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            dep,binding,plan,train=self.parent_fixture(d)
            root=Path(dep['parent_run_root'])
            before={str(p):file_sha256(p) for p in root.rglob('*') if p.is_file()}
            with patch('ricepo.continuation.verify_checkpoint',return_value={'iteration':0,'policy_version':'p'}) as verify:
                result=parent_run(dep,binding,plan,train)
            self.assertEqual(result['lineage']['global_start'],1)
            self.assertEqual(result['saved']['policy_version'],'p')
            verify.assert_called_once()
            self.assertEqual(before,{str(p):file_sha256(p) for p in root.rglob('*') if p.is_file()})
            (Path(dep['parent_source_root'])/'src/ricepo/original.py').write_text('changed=True')
            with self.assertRaisesRegex(ValueError,'source'):
                parent_run(dep,binding,plan,train)

    def test_changed_reference_or_schedule_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            dep,binding,plan,train=self.parent_fixture(d)
            with self.assertRaisesRegex(ValueError,'model_sha256'):
                parent_run(dep,{**binding,'model_sha256':'another reference'},plan,train)
            with patch('ricepo.continuation.verify_checkpoint',return_value={'iteration':0,'policy_version':'p'}):
                Path(dep['continuation_schedule_path']).write_text('{"batches":[]}')
                with self.assertRaisesRegex(ValueError,'schedule'):
                    parent_run(dep,binding,plan,train)

    def test_only_budget_may_change(self):
        parent = dict(total_updates=1, temperature=1, development_every_updates=3)
        continuation_recipe(parent, {**parent, 'total_updates':2})
        with self.assertRaisesRegex(ValueError, 'temperature'):
            continuation_recipe(parent, {**parent, 'total_updates':2, 'temperature':.6})

    def test_schedule_uses_global_index_and_is_order_independent(self):
        keys = ['q'+str(i) for i in range(128)]
        a = scheduled_queries(keys, 42, 1, 4)
        self.assertEqual(a, scheduled_queries(reversed(keys),42,1,4))
        self.assertNotEqual(a, scheduled_queries(keys,42,0,4))

    def test_parent_and_child_state_positions_are_distinct(self):
        validate_state_position(dict(binding='parent',iteration=0,seed=42,policy_version='p'),
                                binding='parent',iteration=0,seed=42,policy_version='p')
        with self.assertRaisesRegex(ValueError, 'identity'):
            validate_state_position(dict(binding='parent',iteration=0,seed=42,policy_version='p'),
                                    binding='child',iteration=0,seed=42,policy_version='p')

    def test_changed_optimizer_position_rejected(self):
        with self.assertRaisesRegex(ValueError, 'identity'):
            validate_state_position(dict(binding='b',iteration=1,seed=42,policy_version='p'),
                                    binding='b',iteration=0,seed=42,policy_version='p')
