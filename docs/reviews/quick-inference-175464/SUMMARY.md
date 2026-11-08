# Quick vLLM inference screen

This is a pure-inference throughput screen, not a RICE-PO training or paper-reproduction result. It used DeepSeek-R1-Distill-Qwen-1.5B, the frozen native two-stage protocol, temperature 1, top-p 1, top-k disabled, a 32,768-token context limit, and a fixed saved-native request list. No retrieval, credit computation, PPO forward/backward pass, optimizer step, development evaluation, or checkpoint was run.

Candidate B was fastest: BF16, FlashAttention, 144 sequences, 32,768 batched tokens and 0.90 GPU memory utilization. Across two single-A800 runs it averaged 1,913.88 generated token/s. Candidate C is the fastest IEEE FP32 configuration in this screen at 232.30 token/s; D was slower at 216.01 token/s.

The BF16 candidates were independently checked against the existing HF probability tolerance on 16 saved real-prefix outputs. Both A and B failed that threshold. Therefore B is a measured pure-inference option only; it is not PPO-training compatible. C remains the fastest configuration with prior IEEE PPO-compatibility evidence.

The valid four-card B confirmation used four independent TP1 workers with 144 fixed requests each. It completed 576 requests and 138,655 generated tokens in 69.971 seconds of shared wall time, or 1,981.60 token/s. That wall time includes independent model initialization and warm-up; it must not be compared to a sum of the single-card formal-generation rates.

The request set contained short saved prefixes and the longest real RICE history available in saved evidence. No saved prefix exceeded 16,384 tokens, so this does not establish performance for a true 16K+ RICE history. The short 512-token stage cap is a screening workload, not a validation of the normal 4,096-token two-stage action budget.

Private evidence hashes: benchmark CSV `55279a2b2eaf4d83e6361e9c1b1b3179b026a5103de22675436f7014c21c68c0`; repeated ranking `e7530eae9b1cad42350d37fcf30705342c2921ea825355586ab5ca68252c36f0`; probability audit `5b62d6e8ea45db3c7bf5bfc330ee7e23af627f9b7d77515db7af4c4f96a5f424`; best-config receipt `aae01e93e55990f3fb7f57e68c34a5bd78d715a131c7126761f5af81db80beee`; valid four-card result `8a221061c228315f50028e5bf67ef22d556c8649fda29e380734cabe9f9e4360`.
