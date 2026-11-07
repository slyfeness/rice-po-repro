import unittest
from ricepo.hf_runtime import checkpoint_diagnostic_mode

class DiagnosticModeTests(unittest.TestCase):
    def test_explicit_fresh_single_update_only(self):
        plan={'total_updates':1,'checkpoint_selection':'last'}
        self.assertFalse(checkpoint_diagnostic_mode({},plan,parent=None,resume=False))
        d={'stop_after_checkpoint_diagnostic':True}
        self.assertTrue(checkpoint_diagnostic_mode(d,plan,parent=None,resume=False))
        for parent,resume,p in [(True,False,plan),(None,True,plan),
             (None,False,{**plan,'total_updates':2}),
             (None,False,{**plan,'checkpoint_selection':'best_development_macro'})]:
            with self.assertRaises(ValueError):
                checkpoint_diagnostic_mode(d,p,parent=parent,resume=resume)

if __name__=='__main__':unittest.main()
