"""CPU-only cohort binding checks; no tokenizer/model/download required."""
import copy
import unittest
from scripts.validation.compare_bf16_weights import validate_saved_rows
from ricepo.audit import fingerprint
from ricepo.precision_probe import matched_behavior_gate


class SavedBF16CohortTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(request_id="fixed", prefix_token_ids=[11, 12], token_ids=[13, 14],
                        sampled_log_probs=[-1., -2.], prefix_sha256=fingerprint([11, 12]), seed=4, stage="summary")
        self.selection = dict(requests=[dict(request_id="fixed", prefix_sha256=self.row["prefix_sha256"],
            prefix_tokens=2, action_tokens=2, seed=4, stage="summary")])
        self.config = dict(expected_requests=1, expected_sampled_tokens=2)

    def test_native_ids_and_probability_alignment_preserved(self):
        before = copy.deepcopy(self.row)
        self.assertEqual(validate_saved_rows(self.selection, [self.row], self.config), [before])
        self.assertEqual(self.row, before)

    def test_changed_prefix_rejected(self):
        self.row["prefix_token_ids"][0] = 99
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_saved_rows(self.selection, [self.row], self.config)

    def test_missing_probability_rejected(self):
        self.row["sampled_log_probs"].pop()
        with self.assertRaisesRegex(ValueError, "alignment"):
            validate_saved_rows(self.selection, [self.row], self.config)

    def test_duplicate_or_changed_cohort_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_saved_rows(self.selection, [self.row, self.row], self.config)
        self.config["expected_sampled_tokens"] = 3
        with self.assertRaisesRegex(ValueError, "cohort size"):
            validate_saved_rows(self.selection, [self.row], self.config)

    def test_three_frozen_thresholds_are_conjunctive(self):
        thresholds = dict(mean_abs_log_error_max=.02, p99_abs_log_error_max=.10, clip_fraction_max=.01)
        limits = dict(mean_abs_log_error=.02, p99_abs_log_error=.10, clip_fraction=.01)
        self.assertTrue(matched_behavior_gate(limits, thresholds))
        for key in limits:
            with self.subTest(key=key):
                changed = {**limits, key:limits[key]+.000001}
                self.assertFalse(matched_behavior_gate(changed, thresholds))
