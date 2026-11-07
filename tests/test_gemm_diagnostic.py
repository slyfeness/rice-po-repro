import json,math,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from ricepo.artifacts import file_sha256
from ricepo.precision_pilot import pilot_settings
from ricepo.vllm_two_stage import VLLMEngine,vllm_numeric_options


class GemmGateTests(unittest.TestCase):
    def test_explicit_gemm_mode_cannot_silently_use_default_matmul(self):
        self.assertFalse(vllm_numeric_options('tf32_gemm_pilot')['ppo_compatible'])
        with patch.dict('os.environ',{'TRITON_F32_DEFAULT':'tf32','VLLM_FLOAT32_MATMUL_PRECISION':'highest'}):
            with self.assertRaisesRegex(ValueError,'worker matmul'):VLLMEngine('not-loaded',None,'p',numeric_mode='tf32_gemm_pilot')

    def test_B_gate_needs_speed_IEEE_teacher_and_exactly_one_update(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root);plan=p/'plan.json';check=p/'check.json';speed=p/'speed.json'
            plan.write_text(json.dumps(dict(kind='tf32_gemm_B_independent_v1',maximum_updates=1,training_seed=42,
                conditions=dict(query_composition={'mode':'original_and_summary','separator':'\n'}))))
            sha=file_sha256(plan)
            ck=dict(candidate='B',private_plan_sha256=sha,risk_screen_pass=True,teacher_matmul_allow_tf32=False,teacher_precision='highest')
            sp=dict(private_plan_sha256=sha,speedup=1.5);check.write_text(json.dumps(ck));speed.write_text(json.dumps(sp))
            deployment=dict(sampling_backend='vllm_mixed_precision_pilot_v1',stop_after_checkpoint_diagnostic=True,
                precision_pilot=dict(candidate='B',validation_plan_path=str(plan),validation_plan_sha256=sha,
                    heldout_result_path=str(check),speed_gate_evidence_path=str(speed)))
            recipe=dict(method='ricepo',queries_per_update=2,total_updates=1,trajectories_per_query=4,anchors_per_query=4,
                total_branches_including_original=6,rollout_depth=5,training_seeds=[42],query_composition={'mode':'original_and_summary','separator':'\n'},
                local_advantage_normalization='center_only',ppo_epochs=1,clip_epsilon=.2,actor_learning_rate=1e-6,
                actor_kl_coefficient=.001,influence_threshold=.05,residual_threshold=.03)
            with patch('ricepo.artifacts.require_cluster_storage',side_effect=Path):
                self.assertEqual(pilot_settings(deployment,recipe)['numeric_mode'],'tf32_gemm_pilot')
                with self.assertRaises(ValueError):pilot_settings(deployment,{**recipe,'total_updates':2})
                speed.write_text(json.dumps({**sp,'speedup':1.4999}))
                with self.assertRaises(ValueError):pilot_settings(deployment,recipe)
                speed.write_text(json.dumps(sp));check.write_text(json.dumps({**ck,'teacher_matmul_allow_tf32':True}))
                with self.assertRaises(ValueError):pilot_settings(deployment,recipe)


class GemmObserverTests(unittest.TestCase):
    def observe(self,signal):
        import torch
        from ricepo.pilot_gradient import prepare
        model=torch.nn.Linear(2,1,bias=False,dtype=torch.float32)
        opt=torch.optim.SGD(model.parameters(),lr=.1)
        items=[SimpleNamespace(advantages=(signal,),generated_mask=(True,))]
        observer,report=prepare(model,opt,items,candidate_forward=None,ieee_forward=None,device='cpu',recipe={},pilot={'candidate':'B'})
        (model(torch.ones(1,2)).sum()*signal).backward()
        observer('before_clip');opt.step();observer('after_step')
        return report

    def test_observes_real_change_without_reference_backward(self):
        r=self.observe(1.)
        self.assertGreater(r['parameter_delta_l2'],0);self.assertGreater(r['changed_parameter_elements'],0)
        self.assertIsNone(r['reward_gradient_nonzero']);self.assertEqual(r['reference_comparison_optimizer_steps'],0)

    def test_legal_zero_signal_is_not_effective_learning(self):
        r=self.observe(0.)
        self.assertEqual(r['changed_parameter_elements'],0);self.assertEqual(r['candidate_gradient_norm'],0)


if __name__=='__main__':unittest.main()
