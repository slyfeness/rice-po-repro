import unittest
from ricepo.inference_timing import request_timing,NOT_MEASURED
from ricepo.vllm_two_stage import vllm_runtime_options
from ricepo.inference_scheduling import partition_requests
from ricepo.inference_benchmark import metric_deltas


class InferenceTimingTests(unittest.TestCase):
    def test_labelled_counter_is_not_silently_overwritten(self):
        rows=[dict(name='vllm:prompt_tokens_by_source',labels={'source':s},value=v) for s,v in [('cache',10),('compute',20)]]
        self.assertEqual(metric_deltas([],rows)['vllm:prompt_tokens_by_source'],NOT_MEASURED)

    def test_clock_domains_and_intervals(self):
        m=dict(arrival_time=1700000000.,queued_ts=20.,scheduled_ts=22.,first_token_ts=25.,last_token_ts=31.,first_token_latency=5.2)
        r=request_timing(m,4)
        self.assertEqual(r,dict(ttft=5.2,queue=2.,prefill=3.,decode=6.,tpot=2.,core_completion=11.))

    def test_unknown_is_not_inferred_from_tokens(self):
        self.assertTrue(all(x==NOT_MEASURED for x in request_timing(None,100).values()))
        self.assertEqual(request_timing(dict(queued_ts=1.,scheduled_ts=2.,first_token_ts=3.,last_token_ts=3.),1)['tpot'],NOT_MEASURED)

    def test_low_memory_requires_explicit_co_resident_mode(self):
        with self.assertRaises(ValueError):vllm_runtime_options({'gpu_memory_utilization':.45},numeric_mode='tf32_pilot')
        self.assertEqual(vllm_runtime_options({'gpu_memory_utilization':.45},numeric_mode='tf32_pilot',allow_low_memory=True)['gpu_memory_utilization'],.45)

    def test_sticky_partitions_preserve_ids_and_ignore_completion_order(self):
        rows=[dict(request_id=str(i),group_key=str(i//3),prefix_token_ids=[1]*(i+1),remaining_token_budget=4096) for i in range(12)]
        p=partition_requests(rows,4)
        self.assertEqual(sorted(sum(p,[])),sorted(r['request_id'] for r in rows))
        # Within-group input order is immaterial to membership or placement.
        q=partition_requests(list(reversed(rows)),4)
        self.assertEqual([set(s) for s in p],[set(s) for s in q])
        for group in range(4):
            self.assertEqual(sum(bool({str(i) for i in range(group*3,group*3+3)}.intersection(s)) for s in p),1)
        with self.assertRaises(ValueError):partition_requests(rows+[rows[0]],4)
