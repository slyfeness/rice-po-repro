import unittest
from types import SimpleNamespace
from ricepo.hf_data_parallel import length_partition, validate_spec, validate_recipe, rng_state_from_bytes, rng_state_to_bytes

class HFDPContractTests(unittest.TestCase):
    def test_rng_bytes_avoid_tensor_ipc_and_round_trip_exactly(self):
        import torch
        original=torch.tensor([0,17,255,31],dtype=torch.uint8)
        payload=rng_state_to_bytes(original)
        self.assertIsInstance(payload,bytes)
        self.assertIs(rng_state_to_bytes(payload),payload)
        self.assertTrue(torch.equal(rng_state_from_bytes(payload),original))

    def test_partition_covers_unequal_lengths_once(self):
        items=[SimpleNamespace(generated_mask=(False,)+(True,)*n) for n in (9,8,2,1,1)]
        self.assertEqual(length_partition(items),[[0,3,4],[1,2]])

    def test_cpu_exact_sequences_are_pinned_to_rank_zero_before_length_balance(self):
        items=[SimpleNamespace(generated_mask=(False,)+(True,)*n) for n in (6000,90,80,70,60)]
        shards=length_partition(items,cpu_exact_budget={'horizon':5535,'action_tokens':1889})
        self.assertIn(0,shards[0])
        self.assertNotIn(0,shards[1])
        self.assertEqual(sorted(index for shard in shards for index in shard),list(range(len(items))))
    def test_rejects_averaging(self):
        with self.assertRaises(ValueError):
            validate_spec(dict(world_size=2,partition='length_greedy_v1',grad_reduce='mean'))
    def test_exact_contract(self):
        validate_spec(dict(world_size=2,partition='length_greedy_v1',grad_reduce='manual_allreduce_sum'))
    def test_integer_512_full_batch_contract(self):
        recipe=dict(ppo_epochs=1,loss_reduction='sequence_mean',max_grad_norm=1,
                    optimizer_minibatch_size=512,gradient_accumulation=512)
        validate_recipe(recipe)
        validate_recipe(recipe,512)
        with self.assertRaises(ValueError):validate_recipe(recipe,256)
        for key,value in [('ppo_epochs',2),('loss_reduction','token_mean'),('max_grad_norm',2),('gradient_accumulation',True)]:
            with self.subTest(key=key),self.assertRaises(ValueError):validate_recipe({**recipe,key:value},512)

    def test_scoring_rejects_mixed_or_missing_policy_identity_before_cuda(self):
        from ricepo.hf_data_parallel import Replica
        replica=Replica.__new__(Replica)
        replica.options={};replica.actor=object();replica.device='cuda:0'
        for policies in (('',),('p0','p1')):
            trajectories=[SimpleNamespace(policy_version=value) for value in policies]
            with self.subTest(policies=policies),self.assertRaisesRegex(ValueError,'one nonempty frozen policy'):
                replica.score(trajectories,entropy=True)

class HFDPCommitTests(unittest.TestCase):
    def build(self, reject=False, other_sha='after'):
        from ricepo.hf_data_parallel import HFDataParallel
        from ricepo.training import UpdateItem
        from unittest.mock import Mock
        dp=HFDataParallel.__new__(HFDataParallel)
        items=[UpdateItem(str(i),'v',(1,2),(False,True),(0.,1.),(0.,-.5)) for i in range(2)]
        dp.pipe=Mock()
        dp.local=Mock()
        dp.local.behavior={};dp.local.recipe=dict(actor_kl_coefficient=.001,ppo_epochs=1,
            loss_reduction='sequence_mean',max_grad_norm=1,optimizer_minibatch_size='full_batch',gradient_accumulation='full_batch')
        dp.local.backward.return_value=[(0,2.,object(),.1)]
        dp.local.step.return_value=(1.,'after')
        from unittest.mock import Mock
        replies=[[(1,3.,object(),.2)],None,(1.,other_sha),{}]
        def receive():
            command=dp.pipe.send.call_args.args[0][0]
            return None if command=='clear_gradients' else replies.pop(0)
        dp._receive=Mock(side_effect=receive)
        gate=Mock()
        gate.report=dict(ratio=dict(min=.9,max=1.1,mean=1.),outside_clip_fraction=0.,effective_clip_fraction=0.)
        if reject:gate.finish.side_effect=ValueError('risk')
        return dp,items,gate
    def test_failed_full_batch_gate_never_reduces_or_steps(self):
        from unittest.mock import patch
        import sys
        dp,items,gate=self.build(reject=True)
        with patch.dict(sys.modules,{'torch':SimpleNamespace()}),patch('ricepo.policy_identity.identify',return_value=('v',{'state_dict_sha256':'before'})):
            with self.assertRaisesRegex(ValueError,'risk'):
                dp.update(items,policy_version='v',probability_guard=gate)
        self.assertEqual(gate.observe.call_count,2)
        dp.local.reduce.assert_not_called();dp.local.step.assert_not_called()
        dp.local.clear_gradients.assert_called_once_with()
        self.assertEqual(dp.pipe.send.call_args.args[0],('clear_gradients',()))
        self.assertEqual(dp._receive.call_count,2)
    def test_observe_failure_cleans_both_replicas_before_raising(self):
        from unittest.mock import patch
        import sys
        dp,items,gate=self.build()
        gate.observe.side_effect=ValueError('observe')
        with patch.dict(sys.modules,{'torch':SimpleNamespace()}),patch('ricepo.policy_identity.identify',return_value=('v',{'state_dict_sha256':'before'})):
            with self.assertRaisesRegex(ValueError,'observe'):
                dp.update(items,policy_version='v',probability_guard=gate)
        dp.local.reduce.assert_not_called();dp.local.step.assert_not_called()
        dp.local.clear_gradients.assert_called_once_with()
        self.assertEqual(dp.pipe.send.call_args.args[0],('clear_gradients',()))
        self.assertEqual(dp._receive.call_count,2)
    def test_hash_mismatch_refuses_commit(self):
        from unittest.mock import patch
        import sys
        dp,items,gate=self.build(other_sha='different')
        with patch.dict(sys.modules,{'torch':SimpleNamespace()}),patch('ricepo.policy_identity.identify',return_value=('v',{'state_dict_sha256':'before'})):
            with self.assertRaisesRegex(ValueError,'SHA differs'):
                dp.update(items,policy_version='v',probability_guard=gate)
    def test_unequal_shards_keep_global_sequence_denominator(self):
        from unittest.mock import patch
        import sys
        dp,items,gate=self.build()
        with patch.dict(sys.modules,{'torch':SimpleNamespace()}),patch('ricepo.policy_identity.identify',return_value=('v',{'state_dict_sha256':'before'})):
            result=dp.update(items,policy_version='v',probability_guard=gate)
        self.assertEqual(result['normalized_loss'],2.5)
        self.assertTrue(result['rank_weight_hash_equal'])
        self.assertTrue(result['actual_state_dict_changed'])
    def test_integer_512_recipe_reaches_complete_update(self):
        from dataclasses import replace
        from unittest.mock import patch
        import sys
        dp,items,gate=self.build()
        items=[replace(items[0],trajectory_id=str(i)) for i in range(512)]
        dp.local.recipe.update(optimizer_minibatch_size=512,gradient_accumulation=512)
        dp.local.backward.return_value=[(i,2.,object(),.1) for i in range(0,512,2)]
        replies=iter([[(i,2.,object(),.1) for i in range(1,512,2)],None,(1.,'after'),{}])
        dp._receive.side_effect=lambda:next(replies)
        with patch.dict(sys.modules,{'torch':SimpleNamespace()}),patch('ricepo.policy_identity.identify',return_value=('v',{'state_dict_sha256':'before'})):
            result=dp.update(items,policy_version='v',probability_guard=gate)
        self.assertEqual(gate.observe.call_count,512)
        self.assertEqual(result['global_sequences'],512)
        self.assertEqual(result['normalized_loss'],2.)
        dp.local.step.assert_called_once_with()
    def test_local_backward_failure_drains_child_before_cleanup_ack(self):
        from unittest.mock import patch
        import sys
        dp,items,gate=self.build()
        dp.local.backward.side_effect=ValueError('backward')
        commands=[]
        def receive():
            command=dp.pipe.send.call_args.args[0][0]
            commands.append(command)
            return [] if command=='backward' else None
        dp._receive.side_effect=receive
        with patch.dict(sys.modules,{'torch':SimpleNamespace()}),patch('ricepo.policy_identity.identify',return_value=('v',{'state_dict_sha256':'before'})):
            with self.assertRaisesRegex(ValueError,'backward'):
                dp.update(items,policy_version='v',probability_guard=gate)
        self.assertEqual(commands,['backward','clear_gradients'])
        dp.local.clear_gradients.assert_called_once_with()
        dp.local.reduce.assert_not_called();dp.local.step.assert_not_called()
