import unittest
from ricepo.training_validation import validate_contract,update_risk_limit,validate_committed_progress
from ricepo.branch_audit import execute_branch_checks
from types import SimpleNamespace

class TrainingValidationTests(unittest.TestCase):
    def recipe(self):
        return dict(method='ricepo',queries_per_update=4,total_updates=3,trajectories_per_query=4,anchors_per_query=4,
            total_branches_including_original=6,rollout_depth=5,training_seeds=[20261107],local_advantage_normalization='center_only',
            ppo_epochs=1,clip_epsilon=.2,actor_learning_rate=1e-6,actor_kl_coefficient=.001,influence_threshold=.05,
            residual_threshold=.03,checkpoint_every_updates=2,development_every_updates=4,checkpoint_selection='last',
            query_composition={'mode':'original_and_summary','separator':'\n'},max_new_tokens_per_step=4096,max_context_tokens=32768,
            temperature=1.,top_p=1.,top_k=0,loss_reduction='sequence_mean')
    def test_contract_and_fixed_budgets(self):
        r=self.recipe();f=dict(kind='tf32_gemm_training_validation_v1',training_seed=20261107,query_batches=[[str(j) for j in range(i*4,i*4+4)] for i in range(3)])
        validate_contract('G_B',f,r)
        with self.assertRaises(ValueError):validate_contract('G_B',f,{**r,'total_updates':128})
        with self.assertRaises(ValueError):validate_contract('G_B',f,{**r,'actor_learning_rate':2e-6})
        with self.assertRaises(ValueError):validate_contract('G_A',f,r)
        self.assertEqual(update_risk_limit(0),.01);self.assertEqual(update_risk_limit(2),.02)
    def test_resume_does_not_replay_a_committed_step(self):
        record=dict(completed_updates=1,policy_version='p',checkpoint_complete_sha256='c',post_update_risk_sha256='r')
        validate_committed_progress(record,index=0,policy='p',checkpoint='c',risk='r')
        with self.assertRaises(ValueError):validate_committed_progress(record,index=1,policy='p',checkpoint='c',risk='r')
    def test_existing_expansion_accepts_registered_risk_callback(self):
        rows=[SimpleNamespace(trajectory_id=str(i)) for i in range(3)];calls=[];events=[]
        def score(t):calls.append(t.trajectory_id);return {'risk_pass':t.trajectory_id!='0','within_bf16_tolerance':False}
        with self.assertRaises(ValueError):
            execute_branch_checks(rows,rows[:1],score=score,save_check=lambda *a:None,save_expansion=lambda s,r:events.append(s),passed=lambda c:c['risk_pass'])
        self.assertEqual(calls,['0','1','2']);self.assertEqual(events,['started','complete'])

if __name__=='__main__':unittest.main()
