import os,unittest
from unittest.mock import patch
from ricepo.scale_memory import validate_memory_repair
from ricepo.scale_training import run_tag
from ricepo.vllm_lifecycle import isolate_hf_allocator


class MemoryScope(unittest.TestCase):
    def test_unknown_memory_choices_rejected(self):
        for value in ({},True,{'mode':'bf16'},{'mode':'cpu_exact_v1'}):
            with self.assertRaises(ValueError):validate_memory_repair(value,'a'*64)

    def test_recovery_alias_keeps_project_stage_and_method(self):
        with patch.dict(os.environ,{},clear=True):self.assertEqual(run_tag('training','grpo'),'biology-scale-v1-training-grpo')
        with patch.dict(os.environ,{'RICE_SCALE_RUN_TAG':'biology-scale-v2-training-grpo'}):
            self.assertEqual(run_tag('training','grpo'),'biology-scale-v2-training-grpo')
            for stage,method in [('capacity','grpo'),('training','ricepo')]:
                with self.assertRaises(ValueError):run_tag(stage,method)
        with patch.dict(os.environ,{'RICE_SCALE_RUN_TAG':'../../another-project'}):
            with self.assertRaises(ValueError):run_tag('training','grpo')

    def test_hf_allocator_isolation_preserves_numeric_worker_settings(self):
        env={'PYTORCH_ALLOC_CONF':'expandable_segments:True','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True',
             'TRITON_F32_DEFAULT':'tf32','VLLM_FLOAT32_MATMUL_PRECISION':'high'}
        with patch.dict(os.environ,env,clear=True):
            isolate_hf_allocator(False);self.assertIn('PYTORCH_ALLOC_CONF',os.environ)
            isolate_hf_allocator(True);self.assertNotIn('PYTORCH_ALLOC_CONF',os.environ)
            self.assertNotIn('PYTORCH_CUDA_ALLOC_CONF',os.environ)
            self.assertEqual(os.environ['TRITON_F32_DEFAULT'],'tf32')
            self.assertEqual(os.environ['VLLM_FLOAT32_MATMUL_PRECISION'],'high')
