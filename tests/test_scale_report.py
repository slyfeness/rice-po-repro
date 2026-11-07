import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('scale_report',Path(__file__).resolve().parents[1]/'scripts/scale_training/report.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


class ReportTests(unittest.TestCase):
    def test_stream_preserves_unicode_and_checks_completed_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'native.json';rows=[dict(id=i,text='中文'+('a'*80000)) for i in range(20)]
            p.write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
            self.assertEqual(list(report.array_items(p,report.digest(p))),rows)
            with self.assertRaises(ValueError):list(report.array_items(p,'0'*64))
    def test_branch_reused_history_is_input_only_and_failure_has_no_tool(self):
        def action(depth,prefix,spans):
            return dict(depth=depth,prefix_token_ids=[1]*prefix,generation=dict(protocol='two_stage_env_summary_v1',
                stages=[dict(token_span=dict(start=a,end=b)) for a,b in spans]))
        trace=dict(steps=[action(1,10,[(0,3),(5,7)])],failure=action(3,20,[(0,4)]))
        self.assertEqual(report.generation_work(trace),(9,45,1))
        self.assertEqual(report.generation_work(trace,2),(4,20,0))


if __name__=='__main__':unittest.main()
