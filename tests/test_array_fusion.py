import unittest
import random
from ricepo.retrieval import FixedHybridRetriever, array_hybrid_rank, hybrid_rank, minmax


class FusionTests(unittest.TestCase):
    def test_exact_reference_with_ties_exclusions_and_constant_scores(self):
        rng = random.Random(91)
        ids = tuple(f"d{i:04d}" for i in range(100))
        for constant in (False, True):
            dense = {key: 1.0 if constant else rng.random() for key in ids}
            sparse = {key: float(rng.randrange(4)) for key in ids}
            for before in (False, True):
                kwargs = dict(excluded_ids=frozenset(ids[::9]), exclude_before_normalization=before, top_k=10)
                expected = hybrid_rank(dense, sparse, normalization=minmax, **kwargs)
                actual = array_hybrid_rank(dense, sparse, ordered_ids=ids, **kwargs)
                self.assertEqual(actual, expected)

    def test_retriever_array_route_preserves_rendered_observation_and_ids(self):
        kwargs = dict(dense_score=lambda q: {'a':1., 'b':2.}, sparse_score=lambda q: {'a':2., 'b':1.},
                      corpus={'a':'first','b':'second'}, normalization=minmax,
                      exclude_before_normalization=True, top_k=2,
                      render_observation=lambda docs: tuple(len(d.text) for d in docs), backend_version='frozen')
        reference = FixedHybridRetriever(**kwargs)
        optimized = FixedHybridRetriever(**kwargs, ranking_impl='array_minmax_v1')
        self.assertEqual(reference.retrieve('q',excluded_ids=frozenset()),
                         optimized.retrieve('q',excluded_ids=frozenset()))


if __name__ == "__main__":
    unittest.main()
