import unittest
from scripts.gpu_probe_common import sub64_indices,partition,splits

class ProbeTests(unittest.TestCase):
    def test_selection_and_partition(self):
        xs=[dict(trajectory_id=str(i),generated_mask=[False]+[True]*i) for i in range(1,513)]
        ids=sub64_indices(xs)
        self.assertEqual(len(set(ids)),64);self.assertIn(511,ids)
        self.assertEqual(ids,sub64_indices(xs))
        shards=partition([xs[i] for i in ids],2)
        self.assertEqual(sorted(shards[0]+shards[1]),list(range(64)))
        self.assertEqual([len(s) for s in splits(64)],[32]*5)
    def test_manual_sum_global_denominator(self):
        import torch
        x=torch.tensor([1.,2.,4.,5.,8.]);p=torch.tensor(2.,requires_grad=True)
        (p*x).square().sum().div(5).backward();ref=p.grad.clone()
        gradients=[]
        for shard in ([0,3],[1,2,4]):
            q=torch.tensor(2.,requires_grad=True)
            for i in shard:((q*x[i]).square()/5).backward()
            gradients.append(q.grad)
        self.assertTrue(torch.allclose(sum(gradients),ref))
if __name__=='__main__':unittest.main()
