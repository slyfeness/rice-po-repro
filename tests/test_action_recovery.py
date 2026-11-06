"""Recovery consumes native evidence, never a successful-seed retry loop."""
import unittest,tempfile
from pathlib import Path
from dataclasses import asdict
from ricepo.action_recovery import VerifiedActionRecovery
from ricepo.artifacts import atomic_json,file_sha256
from ricepo.audit import fingerprint
from ricepo.two_stage import TwoStageFormat
from test_two_stage_format import Tok,MissingTokenTok


class RecoveryTests(unittest.TestCase):
    def fixture(self,d):
        root=Path(d)/'failed';root.mkdir();code=Path(d)/'code/src/ricepo';code.mkdir(parents=True)
        (code/'a.py').write_text('immutable=True')
        sources={'a.py':file_sha256(code/'a.py')}
        config={'fixture':True};plan={'total_updates':2}
        binding=dict(config_sha256=fingerprint(config),recipe_fingerprint='recipe',source_sha256=fingerprint(sources),
            dataset_receipt_sha256='dataset',manifest_sha256={'train':'train','development':'dev','test':'test'},
            model_sha256='model',retriever_sha256='retrieval',prompt_sha256='prompt',behavior={'precision':'float32'},
            continuation={'global_start':1,'parent_checkpoint_complete_sha256':'cp','parent_policy_version':'p','schedule_sha256':'schedule'})
        for name,value in [('config',config),('recipe',{'values':plan}),('binding',binding),('sources',sources)]:
            atomic_json(root/(name+'.json'),value)
        f=TwoStageFormat(Tok());a=tuple(b'r</think>');b=tuple(b's</summary>')
        g=f.record(a,b,(-1.,)*len(a),(-2.,)*len(b),'think_closed','summary_closed')
        events=root/'attempts/first/events'
        first=dict(seed=42,policy_version='p',prefix_sha256=fingerprint((1,2)),query_id='q',trajectory_id='t',depth=1)
        failed={**first,'seed':43,'prefix_sha256':fingerprint((7,)),'trajectory_id':'branch'}
        atomic_json(events/'action.json',dict(kind='two_stage_action_record',generation=asdict(g),**first))
        atomic_json(events/'think.json',dict(kind='two_stage_stage_saved',stage_name='think',native_token_ids=a,
            sampled_log_probs=(-1.,)*len(a),rng_before_sha256='before',rng_after_sha256='after',**failed))
        for n,identity in enumerate((first,failed)):
            atomic_json(root/f'requests/{n}/first.json',dict(attempt='first',identity=identity))
        atomic_json(root/'attempts/first/failure.json',dict(reason='unmapped native byte token',operation=failed))
        return root,binding,plan,g,a

    def test_saved_action_is_native_and_prefix_bound(self):
        with tempfile.TemporaryDirectory() as d:
            root,binding,plan,g,a=self.fixture(d)
            recovery=VerifiedActionRecovery([{'run':str(root),'source':str(Path(d)/'code')}],binding,plan)
            self.assertEqual(recovery((1,2),42,'p'),g)
            self.assertTrue(recovery.paid_reexecution((7,),43,'p'))
            self.assertIsNone(recovery((8,),99,'p'))
            with self.assertRaisesRegex(ValueError,'prefix'):
                recovery((9,),42,'p')

    def test_failed_request_cannot_turn_into_a_successful_resample(self):
        with tempfile.TemporaryDirectory() as d:
            root,binding,plan,g,a=self.fixture(d)
            recovery=VerifiedActionRecovery([{'run':str(root),'source':str(Path(d)/'code')}],binding,plan)
            with self.assertRaisesRegex(ValueError,'terminal cause'):
                recovery.validate_action((7,),43,'p',g)
            f=TwoStageFormat(MissingTokenTok())
            bad=f.record(a,(1001,),(-1.,)*len(a),(-2.,),'think_closed','native_token_unmapped')
            recovery.validate_action((7,),43,'p',bad)
            with self.assertRaisesRegex(ValueError,'native stage'):
                recovery.validate_stage((7,),43,'p','think',(3,),(-1.,),'before','after')
            recovery.validate_stage((7,),43,'p','think',a,(-1.,)*len(a),'before','after')

    def test_wrong_assets_or_already_committed_update_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root,binding,plan,g,a=self.fixture(d)
            sources=[{'run':str(root),'source':str(Path(d)/'code')}]
            with self.assertRaisesRegex(ValueError,'model_sha256'):
                VerifiedActionRecovery(sources,{**binding,'model_sha256':'different'},plan)
            atomic_json(root/'iteration-000000/optimizer_step_committed.receipt.json',{})
            with self.assertRaisesRegex(ValueError,'committed'):
                VerifiedActionRecovery(sources,binding,plan)
