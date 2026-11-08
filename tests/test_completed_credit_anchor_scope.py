import copy,unittest
from dataclasses import replace,asdict
from test_failure_zero import runner,rollout
from ricepo.completed_credit_recovery import reselect_anchors
from ricepo.credit import select_anchors


class RecoveredAnchorScope(unittest.TestCase):
    def test_same_query_pooling_with_global_saved_entropy_table(self):
        a=replace(rollout(runner(),"a"),query_id="q1")
        b=replace(rollout(runner(),"b"),query_id="q2")
        groups={"q1":[a],"q2":[b]};entropy={"a":[1.]*len(a.token_ids),"b":[100.]*len(b.token_ids)}
        before=copy.deepcopy([asdict(a),asdict(b),entropy])
        expected=select_anchors([a],{"a":entropy["a"]},top_k=2)+select_anchors([b],{"b":entropy["b"]},top_k=2)
        self.assertEqual(reselect_anchors(groups,entropy,2),expected)
        self.assertEqual([asdict(a),asdict(b),entropy],before)
        self.assertEqual([x.query_id for x in expected],["q1","q1","q2","q2"])

    def test_no_summary_does_not_create_an_anchor(self):
        a=rollout(runner((1,)),"failed")
        self.assertEqual(reselect_anchors({"q":[a]},{"failed":[float("nan")]*len(a.token_ids)},4),())
