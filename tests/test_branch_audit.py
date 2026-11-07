import unittest
from types import SimpleNamespace
from ricepo.branch_audit import branch_check_plan, select_branch_checks


class BranchAuditTests(unittest.TestCase):
    def test_first_update_exhaustive_and_later_order_independent(self):
        rows = [SimpleNamespace(trajectory_id=f"b{i}", token_ids=(1,) * 5, steps=(1,), failure=None)
                for i in range(100)]
        self.assertEqual(len(select_branch_checks(rows, update_index=0, policy="stratified_five_percent_v1")), 100)
        a = select_branch_checks(rows, update_index=1, policy="stratified_five_percent_v1")
        b = select_branch_checks(rows[::-1], update_index=1, policy="stratified_five_percent_v1")
        self.assertEqual([t.trajectory_id for t in a], [t.trajectory_id for t in b])
        self.assertEqual(len(a), 5)
        plan = branch_check_plan(rows, update_index=1, policy="stratified_five_percent_v1")
        self.assertEqual((plan['total'],plan['checked']), (100,5))
        self.assertEqual(plan['strata'], [dict(length='short',stage='summary_present',failure_category='valid',total=100,selected=5)])

    def test_failed_long_branch_stratum_is_retained(self):
        a = SimpleNamespace(trajectory_id="valid", token_ids=(1,), steps=(1,), failure=None)
        b = SimpleNamespace(trajectory_id="failed", token_ids=(1,) * 17000, steps=(),
                            failure=SimpleNamespace(category="length"))
        self.assertEqual(len(select_branch_checks([a, b], update_index=2, policy="stratified_five_percent_v1")), 2)


if __name__ == "__main__":
    unittest.main()
