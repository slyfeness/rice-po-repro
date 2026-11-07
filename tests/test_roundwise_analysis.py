import unittest
from ricepo.roundwise_analysis import summarize, paired_divergence


class RoundwiseTests(unittest.TestCase):
    def row(self,q,d0,scores,failed=False):
        return dict(query_id=q,d0=d0,rounds=[dict(depth=i+1,ndcg=s,summary_sha256=str(s),ranking_sha256=str(s))
                    for i,s in enumerate(scores)],failed=failed,last_valid_ndcg=scores[-1] if scores else d0,
                    task_score=0 if failed else scores[-1])

    def test_matched_denominators_and_failure_decomposition(self):
        out=summarize([self.row('a',1,[.8,.4]),self.row('b',.2,[],True)])
        self.assertAlmostEqual(out['complete_only']['matched_d0'],1)
        self.assertAlmostEqual(out['complete_only']['final_ndcg'],.4)
        self.assertAlmostEqual(out['d0_minus_task'],out['d0_minus_last_valid']+out['failure_utility_gap'])
        self.assertEqual(out['depths'][0]['count'],1)
        self.assertEqual(out['depths'][0]['matched_d0'],1)
        self.assertEqual(out['intermediate_better_than_final_count'],1)

    def test_failures_are_not_retrieval_zero(self):
        out=summarize([self.row('a',1,[.8],True)])
        self.assertEqual(out['mean_last_valid_ndcg'],.8)
        self.assertEqual(out['mean_task_score'],0)
        self.assertIsNone(out['complete_only']['final_ndcg'])

    def test_paired_identity_not_positional_zip(self):
        a=[self.row('a',1,[.5]),self.row('b',1,[1])]
        b=[self.row('b',1,[1]),self.row('a',1,[.2])]
        result=paired_divergence(a,b)
        self.assertEqual(result[0]['query_id'],'a')
        self.assertEqual(result[0]['first_summary_difference_depth'],1)
        with self.assertRaises(ValueError):paired_divergence(a,b[:1])

    def test_duplicate_or_incomplete_identity_rejected(self):
        row=self.row('a',1,[1])
        with self.assertRaises(ValueError):summarize([row,row])
