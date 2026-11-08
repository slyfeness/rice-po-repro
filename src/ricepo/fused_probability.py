"""Opt-in complete-original-batch risk gate; q remains the PPO denominator."""
from .precision_pilot import mismatch_statistics, risk_screen


class FusedOriginalProbabilityGuard:
    def __init__(self, *, outside_max, branch_gate, save_report):
        self.outside_max = outside_max
        self.branch_gate = branch_gate
        self.save_report = save_report

    def begin(self, items, policy_version):
        self.expected = [(x.trajectory_id, x.policy_version) for x in items]
        if any(v != policy_version for _, v in self.expected):
            raise ValueError('wrong behavior version')
        self.seen, self.q, self.p, self.a = [], [], [], []

    def observe(self, item, current):
        if tuple(current.shape) != (1, len(item.token_ids)):
            raise ValueError('native score alignment changed')
        identity = (item.trajectory_id, item.policy_version)
        if len(self.seen) >= len(self.expected) or identity != self.expected[len(self.seen)]:
            raise ValueError('missing, duplicate or reordered originals')
        scores = current.detach()[0].cpu().tolist()
        for q, p, a, mask in zip(item.old_log_probs, scores, item.advantages, item.generated_mask, strict=True):
            if mask:
                self.q.append(q); self.p.append(p); self.a.append(a)
        self.seen.append(identity)

    def finish(self):
        if self.seen != self.expected:raise ValueError('incomplete original batch')
        self.report = mismatch_statistics(self.q, self.p, self.a)
        self.save_report(self.report)
        if not risk_screen(self.report, outside_max=self.outside_max):
            raise ValueError('fused original probability risk failed; no optimizer step')
        # Existing branch checking/expansion must have completed before commit.
        self.branch_gate()
