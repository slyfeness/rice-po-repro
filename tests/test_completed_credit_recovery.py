import copy,unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from test_failure_zero import runner,rollout
from ricepo.completed_credit_recovery import decode_trajectory,reject_committed_source


class CompletedCreditRecovery(unittest.TestCase):
    def test_native_success_and_failure_roundtrip_without_tokenizer(self):
        for r in (runner(),runner((1,)),runner((2,))):
            original=rollout(r);record=asdict(original)
            # Match the JSON container types actually on disk.
            import json
            raw=json.loads(json.dumps(record));saved=copy.deepcopy(raw)
            restored=decode_trajectory(raw,5)
            self.assertEqual(restored,original);self.assertEqual(raw,saved)
            bad=copy.deepcopy(raw);bad['generated_mask'][-1]=not bad['generated_mask'][-1]
            with self.assertRaises(ValueError):decode_trajectory(bad,5)

    def test_a_committed_or_completed_source_cannot_be_replayed(self):
        with TemporaryDirectory() as d:
            root=Path(d);(root/'iteration-000000').mkdir()
            reject_committed_source(root)
            (root/'iteration-000000/optimizer_step_committed.receipt.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'committed'):reject_committed_source(root)
