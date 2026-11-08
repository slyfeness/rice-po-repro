import copy,importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('capacity_completion',Path(__file__).parents[1]/'scripts/scale_training/complete_capacity.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class CapacityCompletion(unittest.TestCase):
    def test_native_prefix_and_complete_groups_required(self):
        rows=[dict(query_id=str(q),trajectory_id=f'{q}-{t}',steps=[dict(prefix_token_ids=[10,20,30])]) for q in range(32) for t in range(4)]
        before=copy.deepcopy(rows)
        self.assertEqual(module.checked_prefix(rows),(10,20,30));self.assertEqual(rows,before)
        for bad in (rows[:-1],rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):module.checked_prefix(bad)
        rows[-1]['query_id']='wrong'
        with self.assertRaises(ValueError):module.checked_prefix(rows)

    def test_failed_first_action_uses_its_saved_native_prefix(self):
        rows=[dict(query_id=str(q),trajectory_id=f'{q}-{t}',steps=[],failure=dict(prefix_token_ids=[10,11])) for q in range(32) for t in range(4)]
        self.assertEqual(module.checked_prefix(rows),(10,11))

    def test_completion_has_no_optimizer_or_retrieval_call(self):
        import ast
        tree=ast.parse(Path(spec.origin).read_text(encoding='utf8'))
        called={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertFalse({'step','backward','retrieve','encode'}&called)
