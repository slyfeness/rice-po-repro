import unittest
import importlib.util
from pathlib import Path
from unittest.mock import patch

from ricepo.vllm_lifecycle import DualTransport, affine_partitions
from ricepo.hf_runtime import _validate_hf_dp_resource_topology


class DualEngineSchedulingTests(unittest.TestCase):
    def test_dp_with_isolated_diver_fails_closed_before_model_loading(self):
        deployment={"hf_data_parallel": {"world_size": 2}, "retrieval_process": "isolated_v1"}
        with self.assertRaisesRegex(ValueError, "CPU residency"):
            _validate_hf_dp_resource_topology(deployment, object(), object())

    def test_dp_with_isolated_diver_accepts_explicit_hf_cpu_residency(self):
        deployment={"hf_data_parallel": {"world_size": 2}, "retrieval_process": "isolated_v1",
                    "retrieval_hf_residency": "cpu_v1"}
        _validate_hf_dp_resource_topology(deployment, object(), object())

    def test_remote_retriever_records_cpu_transition_before_hf(self):
        from ricepo.remote_retrieval import RemoteRetrievalService
        from unittest.mock import Mock
        service=RemoteRetrievalService.__new__(RemoteRetrievalService)
        service.connection=Mock();service.transitions=[]
        service.receive=Mock(return_value=('residency',{'phase':'hf','device':'cpu','seconds':1.25}))
        result=service.set_residency('hf')
        self.assertEqual(result['device'],'cpu')
        self.assertEqual(service.transitions,[{'phase':'hf','device':'cpu','seconds':1.25}])
        self.assertEqual(service.connection.send.call_args.args[0],('residency','hf'))

    def test_v2_smoke_has_its_own_nonresumable_run_namespace(self):
        from ricepo.scale_training import run_tag
        with patch.dict('os.environ',{},clear=True):
            self.assertEqual(run_tag('smoke','ricepo','biology_scale_training_v2'),
                             'biology-scale-v2-smoke-ricepo')
        with patch.dict('os.environ',{'RICE_SCALE_RUN_TAG':'biology-scale-v2-smoke-r1-ricepo'},clear=True):
            self.assertEqual(run_tag('smoke','ricepo','biology_scale_training_v2'),
                             'biology-scale-v2-smoke-r1-ricepo')

    def test_v2_worker_keeps_topology_metadata_out_of_recipe_resolution(self):
        from ricepo.config import load_config, resolve
        root=Path(__file__).resolve().parents[1]
        spec=importlib.util.spec_from_file_location('v2_worker',root/'scripts/scale_training/worker.py')
        worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
        config=load_config(root/'configs/biology_scale_v2_ricepo.json')
        requirements=worker.configure_v2_recipe(config,'smoke')
        config['runtime_bindings'].update(train_split_manifest='private/train.jsonl',development_split_manifest='private/development.jsonl')
        report=resolve(config,'train')
        self.assertEqual(report['issues'],[])
        self.assertEqual((report['values']['queries_per_update'],report['values']['optimizer_minibatch_size'],
                          report['values']['gradient_accumulation'],report['values']['total_updates']),(4,16,16,1))
        self.assertEqual(requirements['retrieval_hf_residency'],'cpu_v1')
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
