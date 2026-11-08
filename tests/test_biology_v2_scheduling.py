import unittest
from unittest.mock import patch

from ricepo.vllm_lifecycle import DualTransport, affine_partitions
from ricepo.hf_runtime import _validate_hf_dp_resource_topology


class DualEngineSchedulingTests(unittest.TestCase):
    def test_dp_with_isolated_diver_fails_closed_before_model_loading(self):
        deployment={"hf_data_parallel": {"world_size": 2}, "retrieval_process": "isolated_v1"}
        with self.assertRaisesRegex(ValueError, "fail-closed"):
            _validate_hf_dp_resource_topology(deployment, object(), object())
    def test_anchor_group_is_never_split_and_coverage_is_exact(self):
        requests = [
            {"prefix": (1,) * 50, "budget": 100},
            {"prefix": (1,) * 50, "budget": 100},
            {"prefix": (2,) * 20, "budget": 100},
            {"prefix": (3,) * 10, "budget": 100},
        ]
        keys = ["anchor:q-d1", "anchor:q-d1", "query:other", "query:last"]
        affinity = {}
        bins = affine_partitions(requests, keys, affinity)
        self.assertEqual(sorted(index for shard in bins for index in shard), [0, 1, 2, 3])
        self.assertTrue({0, 1}.issubset(set(bins[0])) or {0, 1}.issubset(set(bins[1])))
        self.assertIn("anchor:q-d1", affinity)

    def test_existing_group_owner_is_sticky_across_waves(self):
        affinity = {"query:q": 1}
        first = [{"prefix": (1,) * 3, "budget": 7}]
        bins = affine_partitions(first, ["query:q"], affinity)
        self.assertEqual(bins, [[], [0]])
        second = [{"prefix": (1,) * 4, "budget": 7}, {"prefix": (2,), "budget": 7}]
        bins = affine_partitions(second, ["query:q", "query:new"], affinity)
        self.assertIn(0, bins[1])
        self.assertEqual(sorted(index for shard in bins for index in shard), [0, 1])

    def test_second_worker_construction_failure_reaps_the_first_worker(self):
        reaped = []

        class Worker:
            created = 0
            def __init__(self, _spec, wait=False):
                self.index = Worker.created
                Worker.created += 1
                if self.index == 1:
                    raise RuntimeError("second worker failed")
            def _abort(self):
                reaped.append(self.index)

        with patch("ricepo.vllm_lifecycle.WorkerTransport", Worker):
            with self.assertRaisesRegex(RuntimeError, "second worker"):
                DualTransport([{}, {}], {}, object())
        self.assertEqual(reaped, [0])


if __name__ == "__main__":
    unittest.main()
