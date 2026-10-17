"""
Model-specific FLOPS calculation for BailingMoeV2 (Ling-mini).
Architecture:
  - First layer (index 0): Dense MLP (intermediate_size, SwiGLU)
  - Layers 1..N-1: MoE with top-k routing + shared expert
  - GQA attention (4 KV heads, 16 query heads)
  - SwiGLU activations throughout
"""

from typing import List

from mindspeed_llm.fsdp2.utils.flops.flops_base import BaseFlopsEstimator


class BailingMoeV2FlopsEstimator(BaseFlopsEstimator):
    """FLOPS Estimator for BailingMoeV2 (Ling-mini) MoE model"""

    def calculate_achieved_flops(self, tokens_sum: int, batch_seqlens: List[int], delta_time: float) -> float:
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_attention_heads = self.config.num_attention_heads
        num_key_value_heads = self.config.num_key_value_heads
        head_dim = getattr(self.config, "head_dim", hidden_size // num_attention_heads)
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # Dense layer config (first layer)
        intermediate_size = self.config.intermediate_size

        # MoE layer config
        moe_intermediate_size = self.config.moe_intermediate_size
        moe_topk = self.config.num_experts_per_tok
        num_shared_experts = getattr(self.config, "num_shared_experts", 1)
        moe_shared_intermediate = getattr(self.config, "moe_shared_expert_intermediate_size", moe_intermediate_size)

        # First layer is dense, the rest are MoE
        first_k_dense = getattr(self.config, "first_k_dense_replace", 1)
        dense_layers = min(first_k_dense, num_hidden_layers)
        moe_layers = num_hidden_layers - dense_layers

        # Attention projection params (shared by all layers): QKV + output
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)

        # Dense layer: SwiGLU MLP (gate + up + down) → 3 matrices
        dense_mlp_N = hidden_size * intermediate_size * 3

        # MoE layer: gate routing + top-k expert SwiGLU + shared expert SwiGLU
        moe_gate_N = hidden_size * self.config.num_experts  # router projection
        moe_expert_mlp_N = hidden_size * moe_intermediate_size * moe_topk * 3  # top-k experts
        moe_shared_mlp_N = hidden_size * moe_shared_intermediate * num_shared_experts * 3  # shared expert

        # Total non-attention params
        non_attn_params = (
            attn_linear_N * num_hidden_layers
            + dense_mlp_N * dense_layers
            + (moe_gate_N + moe_expert_mlp_N + moe_shared_mlp_N) * moe_layers
            + vocab_size * hidden_size * 2
        )  # embedding + lm_head

        # 6 = forward + backward + weight-gradient (3x for each)
        dense_flops = 6 * non_attn_params * tokens_sum

        # Attention FLOPS (softmax-style: O(n^2 * d))
        seqlen_square_sum = sum(seqlen * seqlen for seqlen in batch_seqlens)
        attn_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        total_flops = dense_flops + attn_flops
        achieved = total_flops * (1.0 / delta_time) / 1e12  # TFLOPS
        return achieved
