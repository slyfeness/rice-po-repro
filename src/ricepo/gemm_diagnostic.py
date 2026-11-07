"""Inspection RPC running inside the actual vLLM GPU worker, not HF training."""


class GemmInspection:
    def gemm_state(self):
        import torch,os
        from collections import Counter
        from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
        model=self.model_runner.model
        methods=Counter(type(m.quant_method).__module__+'.'+type(m.quant_method).__qualname__
            for m in model.modules() if hasattr(m,'quant_method'))
        dispatch=dispatch_unquantized_gemm()
        return dict(pid=os.getpid(),worker_precision_env=os.environ.get('VLLM_FLOAT32_MATMUL_PRECISION'),
            torch_float32_matmul_precision=torch.get_float32_matmul_precision(),
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            triton_f32_default=os.environ.get('TRITON_F32_DEFAULT'),
            linear_dispatch=dispatch.__module__+'.'+dispatch.__qualname__,linear_methods=dict(methods),
            parameter_dtypes=dict(Counter(str(p.dtype) for p in model.parameters())),
            gpu_name=torch.cuda.get_device_name(),cuda_build=torch.version.cuda,
            cache_block_size=self.cache_config.block_size)
