import unittest
import numpy as np
from ricepo.retrieval import DenseVectorScorer, FixedHybridRetriever, minmax


class BatchedRetrievalTests(unittest.TestCase):
    def test_batch_encodes_once_and_keeps_request_exclusions_and_observations(self):
        ids=('c','a','b')
        vectors=np.asarray([[1.,0.],[0.,1.],[1.,1.]],dtype=np.float32)
        query={'x':[1.,0.],'y':[0.,1.]}; calls=[]
        def encode_many(texts):
            calls.append(list(texts));return np.asarray([query[t] for t in texts],dtype=np.float32)
        dense=DenseVectorScorer(ids,vectors,lambda text:query[text],chunk_size=2,encode_queries=encode_many)
        class Sparse:
            def __init__(self):self.ids=ids
            def score_array(self,q):return np.asarray([1.,2.,3.])
            def __call__(self,q):return dict(zip(ids,self.score_array(q)))
        opts=dict(dense_score=dense,sparse_score=Sparse(),corpus={k:k*2 for k in ids},
            normalization=minmax,exclude_before_normalization=True,top_k=2,
            render_observation=lambda docs:tuple(ord(d.document_id) for d in docs),backend_version='fixed')
        reference=FixedHybridRetriever(**opts)
        batch=FixedHybridRetriever(**opts,ranking_impl='array_minmax_v1',query_batch_size=3)
        requests=[dict(query='x',excluded_ids=frozenset({'a'})),dict(query='y',excluded_ids=frozenset({'c'})),
                  dict(query='x',excluded_ids=frozenset({'b'}))]
        expected=tuple(reference.retrieve(r['query'],excluded_ids=r['excluded_ids']) for r in requests)
        self.assertEqual(batch.retrieve_many(requests),expected)
        self.assertEqual(calls,[['x','y','x']])

    def test_batched_cosines_keep_individual_gemv_order(self):
        rng=np.random.default_rng(31)
        values=rng.normal(size=(43,17)).astype('float32')
        query=rng.normal(size=(7,17)).astype('float32')
        dense=DenseVectorScorer(tuple(map(str,range(43))),values,lambda x:query[x],chunk_size=13,
                                encode_queries=lambda xs:query[xs])
        actual=dense.score_many(list(range(7)))
        expected=np.asarray([list(dense(i).values()) for i in range(7)],dtype='float32')
        np.testing.assert_array_equal(actual,expected)


if __name__=='__main__':unittest.main()
