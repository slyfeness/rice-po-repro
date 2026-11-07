"""Focused synthetic tests for the authorized independent failure environment."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from ricepo.contracts import Anchor, Generation, InvalidGeneration, RankedDocument, RetrievalResult, Span
from ricepo.credit import trajectory_advantages
from ricepo.failure_zero import MODE, validate_trajectory, validate_collections
from ricepo.rollout import RolloutRunner, RolloutRules, BranchRequest, MalformedGenerationError
from ricepo.workflow import IterationComponents, run_iteration


class Generator:
    policy_version = "frozen"
    def __init__(self, bad=()):
        self.bad, self.calls = set(bad), []
    def generate(self, prefix_token_ids, *, seed):
        self.calls.append((tuple(prefix_token_ids), seed))
        if seed in self.bad:
            return InvalidGeneration((91,92,99),"eos","tag_structure","missing_summary_close")
        return Generation((11,12),Span(0,1),Span(1,2),"actual summary","stop")


class Retriever:
    def __init__(self): self.calls=[]
    def retrieve(self, query, *, excluded_ids):
        self.calls.append(query)
        return RetrievalResult(query,(RankedDocument("real-doc",1.0),),(80,),"fixed-backend")


def runner(bad=(), strict=False):
    return RolloutRunner(generator=Generator(bad),retriever=Retriever(),
        reward_scorer=SimpleNamespace(score=lambda *a:0.8),
        query_composer=SimpleNamespace(compose=lambda q,s:s),
        rules=RolloutRules(frozenset({"stop"}),20,200,5),
        should_stop=lambda steps:False,final_reward=lambda steps:steps[-1].local_reward,
        failure_semantics="raise" if strict else MODE)


def rollout(r, ident="t", seeds=(1,2,3,4,5)):
    return r.run(query_id="q",trajectory_id=ident,original_query="question",
        initial_prefix_token_ids=(7,),seeds=seeds,old_policy_version="frozen",initial_ndcg=0.6)


class FailureZeroTests(unittest.TestCase):
    def test_first_failure_keeps_all_native_tokens_no_fake_retrieval(self):
        r=runner((1,)); t=rollout(r)
        validate_trajectory(t,5)
        self.assertEqual(t.token_ids,(7,91,92,99))
        self.assertEqual(t.generated_mask,(False,True,True,True))
        self.assertEqual((t.final_reward,t.last_valid_ndcg,t.failure_terminal_reward),(0,0.6,0))
        self.assertEqual((len(r.generator.calls),len(r.retriever.calls),len(t.steps)),(1,0,0))

    def test_later_failure_retains_real_local_score_and_strict_path_rejects(self):
        r=runner((2,)); t=rollout(r)
        validate_trajectory(t,5)
        self.assertEqual((len(t.steps),t.steps[0].local_reward,t.last_valid_ndcg,t.final_reward),(1,0.8,0.8,0))
        with self.assertRaises(MalformedGenerationError): rollout(runner((1,),strict=True))

    def test_four_originals_and_failed_tokens_have_group_advantages(self):
        r=runner((1,))
        ts=[rollout(r,str(n),(1,2,3,4,5) if n<3 else (10,11,12,13,14)) for n in range(4)]
        b=trajectory_advantages(ts,epsilon=1e-8)
        self.assertEqual(len(b.trajectory_ids),4)
        self.assertLess(b.advantages[0][-1],0)
        self.assertGreater(b.advantages[-1][-2],0)
        self.assertTrue(all(row[0]==0 for row in b.advantages))

    def test_five_anchor_failures_remain_undefined_not_zero(self):
        r=runner(range(10,35)); t=rollout(r)
        s=t.steps[0]; a=Anchor("q","t",1,1,s.reasoning_span,s.summary_span)
        c=r.branch(original_query="question",original=t,anchor=a,
            requests=[BranchRequest(str(n),tuple(range(10+n*5,15+n*5))) for n in range(5)])
        validate_collections([c],[a],{"t":t},6,5)
        self.assertEqual(len(c.trajectories),5)
        self.assertTrue(all(o.local_reward is None and o.final_reward==0 for o in c.group.outcomes[1:]))
        self.assertTrue(all(x.failure.prefix_token_ids==s.prefix_token_ids for x in c.trajectories))

    def test_workflow_skips_entire_unestimable_anchor_keeps_original_advantage(self):
        r=runner(range(10,35)); t=rollout(r)
        def branches(t,a):
            return r.branch(original_query="question",original=t,anchor=a,
                requests=[BranchRequest(str(n),tuple(range(10+n*5,15+n*5))) for n in range(5)])
        received=[]
        components=IterationComponents(lambda *a:[t],
            SimpleNamespace(policy_version="frozen",token_entropies=lambda t:[1.0]*len(t.token_ids)),
            branches,lambda ts,b,*a: received.append(b) or {"new_policy_version":"after"},lambda *a:None)
        recipe=dict(format_failure_policy=MODE,method="ricepo",queries_per_update=1,trajectories_per_query=1,
            anchors_per_query=1,total_branches_including_original=6,normalization_epsilon=1e-8,
            anchor_selection="entropy",local_advantage_normalization="center_only",gate_mode="paper",
            influence_threshold=.05,residual_threshold=.03,rollout_depth=5)
        import json
        with TemporaryDirectory() as d:
            run_iteration(components,iteration=0,policy_version="frozen",output_root=d,recipe=recipe,root_seed=42)
            report=json.loads((Path(d)/"iteration-000000/credit-diagnostics.json").read_text())
            self.assertEqual((report["selected_anchor_count"],report["estimable_anchor_count"]),(1,0))
            self.assertEqual(report["fallback"][0]["status"],"not_estimable")
            self.assertEqual(received[0].advantages,trajectory_advantages([t],epsilon=1e-8).advantages)

    def test_later_branch_failure_keeps_anchor_local_measurement(self):
        r=runner((11,)); t=rollout(r)
        s=t.steps[0]; a=Anchor("q","t",1,1,s.reasoning_span,s.summary_span)
        c=r.branch(original_query="question",original=t,anchor=a,requests=[BranchRequest("b",(10,11,12,13,14))])
        validate_collections([c],[a],{"t":t},2,5)
        self.assertEqual((c.group.outcomes[1].local_reward,c.group.outcomes[1].final_reward),(0.8,0))

    def test_evaluation_keeps_failure_in_denominator_and_raw_ndcg_undefined(self):
        import json
        from ricepo.assessment import evaluate_queries
        from ricepo.data import QueryExample
        r=runner()
        r.generator.generate=lambda *args,**kwargs:InvalidGeneration((91,99),"eos","tag_structure","missing summary")
        key='["biology","q"]'
        with TemporaryDirectory() as d:
            summary=evaluate_queries(examples={key:QueryExample(key,"biology","question",frozenset())},
                expected_ids={("biology","q")},split="development",phase="development",method="ricepo",
                root_seed=42,policy_version="frozen",rollout_depth=5,retrievers={"biology":r.retriever},
                runners={"biology":r},renderer=SimpleNamespace(initial=lambda *a:(7,)),reward=r.reward_scorer,
                output=Path(d)/"eval",manifest_sha256="a"*64,backend_fingerprints={"biology":"b"*64})
            rows=json.loads((Path(d)/"eval/results-by-query.json").read_text())
            self.assertEqual((summary["evaluated_total"],summary["expected_total"],summary["macro"]),(1,1,0))
            self.assertEqual(summary["metric"],"failure_adjusted_task_score")
            self.assertEqual((summary["legitimate_complete_fraction"],summary["d0_macro"]),(0,0.8))
            self.assertIsNone(rows[0]["ndcg_at_10"])
            self.assertEqual((rows[0]["last_valid_ndcg"],rows[0]["failure_adjusted_task_score"]),(0.8,0))
            self.assertEqual(summary["last_valid_ndcg_macro"],0.8)
            self.assertIsNone(summary["conditional_final_ndcg"])


if __name__=="__main__": unittest.main()
