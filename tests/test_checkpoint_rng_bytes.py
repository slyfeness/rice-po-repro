import io
import unittest

class CheckpointRngBytesTests(unittest.TestCase):
    def test_rank_rng_digest_survives_weights_only_checkpoint_reload(self):
        import torch
        from ricepo.checkpoint_io import state_digest
        payload={'cpu_rng':torch.random.get_rng_state(),
                 'hf_rank1_rng':{'cpu_rng':bytes(range(256)), 'cuda_rng':b'\x00\xff\x02'}}
        stream=io.BytesIO()
        torch.save(payload,stream)
        stream.seek(0)
        restored=torch.load(stream,weights_only=True)
        self.assertEqual(state_digest(payload),state_digest(restored))
        restored['hf_rank1_rng']['cuda_rng']=b'\x00\xff\x03'
        self.assertNotEqual(state_digest(payload),state_digest(restored))
