# moe-from-scratch
Sparse Mixture-of-Experts in plain PyTorch, with no MoE libraries. It compares top-k, top-k + load-balancing loss and expert-choice routing in a tiny GPT on TinyShakespeare. Finding: capacity caps and expert choice leak future tokens.
