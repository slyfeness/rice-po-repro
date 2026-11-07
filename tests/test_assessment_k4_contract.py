"""Focused K=4 aggregation regression without model or asset execution."""
import unittest
from ricepo.assessment import _conditional_final_ndcg,summarize_assessment


class K4AssessmentTests(unittest.TestCase):
    def test_four_samples_are_query_means_then_domain_macro_not_best_of_n(self):
        rows=[]
        for domain,query,values in [('a','q1',[0.,0.,0.,1.]),('a','q2',[1.,1.,1.,1.]),('b','q3',[.2]*4)]:
            for index,value in enumerate(values):
                rows.append(dict(domain=domain,query_id=query,sample_index=index,ndcg_at_10=value,
                    split='development',method='ricepo',policy_version='frozen',manifest_sha256='a'*64,
                    behavior_sha256='b'*64,retrieval_set_sha256='c'*64,
                    backend_fingerprint='d'*64,backend_version='fixed'))
        result=summarize_assessment(rows,{('a','q1'),('a','q2'),('b','q3')},split='development',
                                    manifest_sha256='a'*64,samples_per_query=4)
        self.assertAlmostEqual(result['macro'],.4125)
        self.assertEqual((result['evaluated_total'],result['evaluated_sample_total']),(3,12))
        with self.assertRaises(ValueError):
            summarize_assessment(rows+rows[:1],{('a','q1'),('a','q2'),('b','q3')},split='development',
                                  manifest_sha256='a'*64,samples_per_query=4)

    def test_conditional_ndcg_preserves_query_statistical_unit(self):
        rows=[dict(domain='a',query_id='q1',failed=False,ndcg_at_10=0.)]
        rows.extend(dict(domain='a',query_id='q2',failed=False,ndcg_at_10=1.) for _ in range(4))
        rows.append(dict(domain='a',query_id='q1',failed=True,ndcg_at_10=None))
        self.assertEqual(_conditional_final_ndcg(rows),.5)


if __name__=='__main__':unittest.main()
