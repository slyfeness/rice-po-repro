import unittest
from types import SimpleNamespace
from ricepo.branch_audit import branch_check_plan, execute_branch_checks, select_branch_checks


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

    def test_numeric_sample_failure_expands_remaining_once_and_always_blocks_update(self):
        rows=[SimpleNamespace(trajectory_id=str(i)) for i in range(5)]
        called=[];saved=[];events=[]
        def score(t):
            called.append(t.trajectory_id)
            return {'within_bf16_tolerance':t.trajectory_id not in {'1','4'}}
        with self.assertRaisesRegex(ValueError,'expanded'):
            execute_branch_checks(rows,rows[:3],score=score,
                save_check=lambda t,c,expanded:saved.append((t.trajectory_id,expanded)),
                save_expansion=lambda stage,record:events.append((stage,record)))
        self.assertEqual(called,['0','1','2','3','4'])
        self.assertEqual(saved,[('0',False),('1',False),('2',True),('3',True),('4',True)])
        self.assertEqual([stage for stage,_ in events],['started','complete'])
        final=events[-1][1]
        self.assertEqual((final['checked'],final['reused_checks'],final['new_expanded_checks']),(5,2,3))
        self.assertEqual(final['failed_trajectories'],['1','4'])
        self.assertFalse(final['optimizer_update_permitted'])

    def test_expanded_infrastructure_failure_records_incomplete_and_stops(self):
        rows=[SimpleNamespace(trajectory_id=str(i)) for i in range(4)]
        called=[];events=[]
        def score(t):
            called.append(t.trajectory_id)
            if t.trajectory_id=='2':raise OSError('device lost')
            return {'within_bf16_tolerance':False}
        with self.assertRaisesRegex(OSError,'device lost'):
            execute_branch_checks(rows,rows[:1],score=score,save_check=lambda *a:None,
                save_expansion=lambda stage,record:events.append((stage,record)))
        self.assertEqual(called,['0','1','2'])
        self.assertEqual([stage for stage,_ in events],['started','failed'])
        self.assertFalse(events[-1][1]['coverage_complete'])
        self.assertFalse(events[-1][1]['optimizer_update_permitted'])

    def test_initial_infrastructure_failure_is_fail_fast_without_fake_expansion(self):
        row=SimpleNamespace(trajectory_id='a');events=[]
        def score(t):raise RuntimeError('CUDA unavailable')
        with self.assertRaisesRegex(RuntimeError,'CUDA unavailable'):
            execute_branch_checks([row],[row],score=score,save_check=lambda *a:None,
                save_expansion=lambda *a:events.append(a))
        self.assertEqual(events,[])


if __name__ == "__main__":
    unittest.main()
