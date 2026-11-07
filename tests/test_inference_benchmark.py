import unittest


def _row(index, *, stage, source_kind, prefix_length):
    return {
        "request_id": f"request-{index}",
        "stage": stage,
        "source_kind": source_kind,
        "prefix_token_ids": list(range(1, prefix_length + 1)),
        "seed": index,
        "remaining_token_budget": 4096,
        "stop_marker": "</think>" if stage == "think" else "</summary>",
    }


def _manifest():
    rows = []
    for index in range(128):
        rows.append(_row(
            index,
            stage=("think" if index % 2 else "summary"),
            source_kind=("original" if index % 3 else "branch"),
            prefix_length=(4096 if index % 3 == 0 else 12000 if index % 3 == 1 else 20000),
        ))
    return rows


class FixedInferenceManifestTests(unittest.TestCase):
    def test_requires_full_real_request_coverage(self):
        from ricepo.inference_benchmark import validate_request_manifest

        summary = validate_request_manifest(_manifest(), expected_count=128)

        self.assertEqual(summary["request_count"], 128)
        self.assertEqual(summary["stages"], {"think": 64, "summary": 64})
        self.assertEqual(summary["source_kinds"], {"branch": 43, "original": 85})
        self.assertEqual(summary["context_bins"], {"short": 43, "medium": 43, "long": 42})

    def test_rejects_duplicate_request_or_missing_coverage(self):
        from ricepo.inference_benchmark import validate_request_manifest

        duplicate = _manifest()
        duplicate[-1]["request_id"] = duplicate[0]["request_id"]
        with self.assertRaisesRegex(ValueError, "duplicate request_id"):
            validate_request_manifest(duplicate, expected_count=128)

        no_branch = _manifest()
        for row in no_branch:
            row["source_kind"] = "original"
        with self.assertRaisesRegex(ValueError, "branch"):
            validate_request_manifest(no_branch, expected_count=128)

    def test_stage_batches_preserve_stage_specific_stop_and_separate_warmup_paths(self):
        from ricepo.inference_benchmark import stage_batches, response_path

        rows = [_row(0, stage="think", source_kind="original", prefix_length=4096),
                _row(1, stage="summary", source_kind="branch", prefix_length=20000)]
        batches = stage_batches(rows)

        self.assertEqual([row["request_id"] for row in batches["think"]], ["request-0"])
        self.assertEqual([row["request_id"] for row in batches["summary"]], ["request-1"])
        self.assertEqual(response_path("responses", "request-0"), "responses/request-0.json")
        self.assertEqual(response_path("warmup-responses", "request-0"), "warmup-responses/request-0.json")
