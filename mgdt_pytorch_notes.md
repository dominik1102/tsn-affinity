# MGDT PyTorch scaffold

This file accompanies `mgdt_pytorch.py`.

What is already implemented:
- MGDT-style sequence order: `[obs patches, return, action, reward]` per step
- patch-based observation embedding for Atari frames
- discrete tokenization of returns-to-go and rewards
- causal transformer with a spatial-causal mask for observation patches
- separate heads for next-return, next-action, and next-reward prediction
- training loss helper
- expert-action-inference primitives (`sample_expert_return`, `sample_action_from_logits`)

What is intentionally left for the next step:
- exact data pipeline matching the Google notebook
- Atari dataset loader for multi-game batches
- rollout-time autoregressive cache / fast inference loop
- data augmentation and optimizer schedule from the original JAX setup
- optional support for `single_return_token` beyond the masking logic already present

Recommended next milestones:
1. Build a `MultiGameAtariDataset` that mixes games in one batch.
2. Match return discretization to your exported Atari datasets.
3. Write an evaluation loop with expert-action inference.
4. Benchmark joint MGDT-5 on your five Atari games.
5. Only then plug CL strategies on top of the MGDT backbone.
