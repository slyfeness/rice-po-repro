import json
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from ricepo.native_replay import SavedNativeCompletions
from ricepo.audit import fingerprint


class NativeReplayTests(unittest.TestCase):
    def test_exact_policy_seed_prefix_and_all_bound_assets_required(self):
        binding={k:k for k in ("config_sha256","recipe_fingerprint","dataset_receipt_sha256","manifest_sha256",
            "model_sha256","retriever_sha256","prompt_sha256","behavior","source_sha256")}
        with TemporaryDirectory() as d, patch("ricepo.native_replay.require_cluster_storage",side_effect=Path):
            root=Path(d); (root/"binding.json").write_text(json.dumps(binding))
            p=root/"attempts/one/events/one.json";p.parent.mkdir(parents=True)
            p.write_text(json.dumps(dict(kind="raw_completion",policy_version="old",seed=10,
                prefix_token_ids=[1,2],prefix_sha256=fingerprint((1,2)),token_ids=[3,4],finish_reason="eos")))
            cache=SavedNativeCompletions([d],binding)
            self.assertEqual(cache((1,2),10,"old")["token_ids"],(3,4))
            self.assertIsNone(cache((1,2),11,"old"))
            self.assertIsNone(cache((1,3),10,"old"))
            self.assertIsNone(cache((1,2),10,"new"))
            with self.assertRaises(ValueError): SavedNativeCompletions([d],{**binding,"model_sha256":"different"})

if __name__=="__main__":unittest.main()
