"""Explicit process-local HF GEMM scopes; no import-time precision changes."""
from contextlib import contextmanager


@contextmanager
def hf_matmul_scope(precision):
    """Restore all touched flags even on error. Use in an isolated HF process.

    PyTorch precision flags are process-global: concurrent scopes in threads are
    unsupported. This scope does not alter vLLM workers, weights or optimizer.
    Kernel execution still needs independent profiler verification.
    """
    if precision not in ('highest', 'high'):
        raise ValueError('HF GEMM precision must be highest or high')
    import torch
    old = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
           torch.backends.cudnn.allow_tf32)
    try:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = precision == 'high'
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.set_float32_matmul_precision(old[0])
        torch.backends.cuda.matmul.allow_tf32 = old[1]
        torch.backends.cudnn.allow_tf32 = old[2]
