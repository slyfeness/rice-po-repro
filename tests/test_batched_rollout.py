import unittest
import tempfile
from pathlib import Path
from ricepo.audit import attempt
from ricepo.batched_rollout import OriginalRequest, run_many
from ricepo.rollout import RolloutRunner,RolloutRules,BranchRequest
from ricepo.contracts import Generation,InvalidGeneration,Span,RetrievalResult,RankedDocument,Anchor


class Generator:
    policy_version='fixed'
    def __init__(self):self.calls=[]
    def generate(self,prefix,*,seed):
        self.calls.append((tuple(prefix),seed))
        if seed==99:return InvalidGeneration((99,),'eos','tag_structure','fixture')
        return Generation((seed,seed+100),Span(0,1),Span(1,2),str(seed),'stop')
    def generate_many(self,prefixes,seeds):
        return tuple(self.generate(p,seed=s) for p,s in zip(prefixes,seeds,strict=True))


class Retriever:
    def retrieve(self,q,*,excluded_ids):
        return RetrievalResult(q,(RankedDocument('doc',.5),),(800,801),'fixed-index')
class Reward:
    def score(self,q,ids):return .5
class Composer:
    def compose(self,original,summary):return original+' '+summary
def runner():
    return RolloutRunner(generator=Generator(),retriever=Retriever(),reward_scorer=Reward(),query_composer=Composer(),
        rules=RolloutRules(frozenset({'stop'}),10,300,5),should_stop=lambda s:False,
        final_reward=lambda s:s[-1].local_reward,failure_semantics='invalid_action_terminal_zero_v1')
def req(n,seeds=(1,2,3,4,5)):
    return OriginalRequest('q','t'+str(n),'question',(41,42),seeds,'fixed',.8)


class BatchedRolloutTests(unittest.TestCase):
    def test_ready_retrieval_wave_keeps_failed_rows_out_of_tool_batch(self):
        class BatchRetriever(Retriever):
            def __init__(self):self.waves=[]
            def retrieve_many(self,requests):
                self.waves.append(requests)
                return tuple(self.retrieve(r['query'],excluded_ids=r['excluded_ids']) for r in requests)
        requests=[req(1,(1,99,3,4,5)),req(2,(6,7,8,9,10))]
        expected=tuple(runner().run(**r.serial_kwargs()) for r in requests)
        r=runner();r.retriever=BatchRetriever()
        self.assertEqual(run_many(r,requests),expected)
        self.assertEqual([len(w) for w in r.retriever.waves],[2,1,1,1,1])
        self.assertEqual(r.retriever.waves[1][0]['query'],'question 7')

    def test_same_transition_and_native_history_as_serial(self):
        requests=[req(1),req(2,(6,7,8,9,10))]
        serial=runner();batch=runner()
        expected=tuple(serial.run(**r.serial_kwargs()) for r in requests)
        self.assertEqual(run_many(batch,requests),expected)
        self.assertEqual(run_many(runner(),list(reversed(requests))),tuple(reversed(expected)))

    def test_invalid_row_retained_without_further_requests(self):
        r=runner();out=run_many(r,[req(1,(1,99,3,4,5)),req(2,(6,7,8,9,10))])
        self.assertEqual(out[0].final_reward,0)
        self.assertEqual(out[0].last_valid_ndcg,.5)
        self.assertEqual(len(out[0].steps),1)
        self.assertEqual(len(out[1].steps),5)
        self.assertEqual(len(r.generator.calls),7)
        self.assertEqual(out[0].token_ids[-1],99)

    def test_batched_branches_use_exact_prefix_and_keep_equal_samples(self):
        r=runner();t=r.run(**req(1).serial_kwargs());s=t.steps[2]
        a=Anchor(t.query_id,t.trajectory_id,3,1,s.reasoning_span,s.summary_span)
        requests=[BranchRequest('b1',(11,12,13)),BranchRequest('b2',(14,15,16))]
        expected=r.branch(original_query='question',original=t,anchor=a,requests=requests)
        actual=r.branch(original_query='question',original=t,anchor=a,requests=requests,batched=True)
        self.assertEqual(actual,expected)
        for branch in actual.trajectories:
            self.assertEqual(branch.steps[2].prefix_token_ids,s.prefix_token_ids)
            self.assertEqual(branch.steps[:2],t.steps[:2])

    def test_cross_anchor_wave_preserves_each_saved_prefix_and_branch_group(self):
        source=runner();first=source.run(**req(1).serial_kwargs());second=source.run(**req(2,(6,7,8,9,10)).serial_kwargs())
        anchors=[]
        for trace,depth in ((first,2),(second,3)):
            step=trace.steps[depth-1]
            anchors.append(Anchor(trace.query_id,trace.trajectory_id,depth,1,step.reasoning_span,step.summary_span))
        jobs=[]
        for trace,anchor,ids in zip((first,second),anchors,(('b11','b12'),('b21','b22')),strict=True):
            jobs.append(dict(original_query='question',original=trace,anchor=anchor,
                requests=[BranchRequest(ids[0],(11,12,13,14)[:6-anchor.depth]),
                          BranchRequest(ids[1],(21,22,23,24)[:6-anchor.depth])]))
        expected=tuple(runner().branch(**job) for job in jobs)
        actual=runner().branch_many(jobs)
        self.assertEqual(actual,expected)
        for collection,anchor in zip(actual,anchors,strict=True):
            self.assertTrue(all(trace.steps[anchor.depth-1].prefix_token_ids == anchor_prefix
                                for trace,anchor_prefix in ((t,collection.group.outcomes[0].prefix_token_ids)
                                                             for t in collection.trajectories)))

    def test_policy_mutation_or_duplicate_request_rejected(self):
        with self.assertRaises(ValueError):run_many(runner(),[req(1),req(1)])
        r=runner();r.generator.policy_version='new'
        with self.assertRaises(Exception):run_many(r,[req(1)])

    def test_duplicate_registered_request_rejected_before_paid_generation(self):
        r=runner()
        with tempfile.TemporaryDirectory() as directory, attempt(Path(directory)):
            run_many(r,[req(1)])
            count=len(r.generator.calls)
            with self.assertRaisesRegex(ValueError,'duplicate'):
                run_many(r,[req(1),req(2,(6,7,8,9,10))])
            self.assertEqual(len(r.generator.calls),count)
