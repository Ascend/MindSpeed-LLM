"""
Model-specific FLOPS calculation for Qwen3 series (Moe + Dense).
Only contains Qwen3-related logic (single responsibility).
"""
from typing import List

from mindspeed_llm.fsdp2.utils.flops.flops_base import BaseFlopsEstimator

# --------------------------
# Qwen3MoeFlopsEstimator
# --------------------------
class Qwen3MoeFlopsEstimator(BaseFlopsEstimator):
    """FLOPS Estimator for Qwen3 MoE (Mixture of Experts) model"""
    def calculate_achieved_flops(self, tokens_sum: int, batch_seqlens: List[int], delta_time: float) -> float:
        """
        Calculate achieved FLOPS for Qwen3 MoE model (TFLOPS)
        Core Formula (Qwen3 MoE):
        Total_FLOPS = (
            # Dense Layer FLOPS (forward + backward + gradient)
            6 * tokens_sum * (
                (
                    hidden_size*moe_num_expert + 
                    hidden_size*moe_intermediate_size*moe_topk*3 + 
                    hidden_size*(q_size+k_size+v_size+num_attention_heads*head_dim)
                ) * num_hidden_layers + 
                vocab_size*hidden_size*2  # Embedding + LM Head
            ) + 
            # Attention Layer FLOPS
            12 * sum(seqlen**2 for seqlen in batch_seqlens) * head_dim * num_attention_heads * num_hidden_layers
        )
        Achieved_FLOPS (TFLOPS) = (Total_FLOPS / delta_time) / 1e12
        
        Where:
        - q_size = num_attention_heads * head_dim, k_size/v_size = num_key_value_heads * head_dim
        - head_dim = hidden_size // num_attention_heads (or config.head_dim if available)
        - sum(seqlen**2 for seqlen in batch_seqlens) = sum of square of each sequence length in batch
        """
        # Extract model hyperparameters from configuration
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        moe_intermediate_size = self.config.moe_intermediate_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        moe_num_expert = self.config.num_experts
        moe_topk = self.config.num_experts_per_tok

        # Calculate head dimension (use config.head_dim if available, else compute)
        head_dim = getattr(self.config, "head_dim", hidden_size // num_attention_heads)
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # Calculate number of parameters for non-attention layers per layer
        moe_gata_N = hidden_size * moe_num_expert  # MoE gate projection
        moe_expertmlp_N = hidden_size * moe_intermediate_size * moe_topk * 3  # SwiGLU (gate+up+down) × top-k
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2  # Embedding + LM head (forward + backward)
        
        # Total non-attention parameters across all layers
        moe_N = (moe_gata_N + moe_expertmlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        dense_N_flops = 6 * moe_N * tokens_sum  # 6× (forward+backward+gradient) × tokens

        # Calculate attention layer FLOPS (sequence length squared term)
        seqlen_square_sum = sum(seqlen * seqlen for seqlen in batch_seqlens)
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # Total FLOPS and convert to TFLOPS
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

# --------------------------
# Qwen3DenseFlopsEstimator
# --------------------------
class Qwen3DenseFlopsEstimator(BaseFlopsEstimator):
    """FLOPS Estimator for Qwen3 dense (non-MoE) model"""
    def calculate_achieved_flops(self, tokens_sum: int, batch_seqlens: List[int], delta_time: float) -> float:
        """
        Calculate achieved FLOPS for Qwen3 Dense model (TFLOPS)
        Core Formula (Qwen3 Dense):
        Total_FLOPS = (
            # Dense Layer FLOPS (forward + backward + gradient)
            6 * tokens_sum * (
                (
                    hidden_size*intermediate_size*3 + 
                    hidden_size*(q_size+k_size+v_size+num_attention_heads*head_dim)
                ) * num_hidden_layers + 
                vocab_size*hidden_size*2  # Embedding + LM Head
            ) + 
            # Attention Layer FLOPS
            12 * sum(seqlen**2 for seqlen in batch_seqlens) * head_dim * num_attention_heads * num_hidden_layers
        )
        Achieved_FLOPS (TFLOPS) = (Total_FLOPS / delta_time) / 1e12
        
        Where:
        - q_size = num_attention_heads * head_dim, k_size/v_size = num_key_value_heads * head_dim
        - head_dim = hidden_size // num_attention_heads
        - sum(seqlen**2 for seqlen in batch_seqlens) = sum of square of each sequence length in batch
        """
        # Extract model hyperparameters from configuration
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size

        # Calculate head dimension
        head_dim = hidden_size // num_attention_heads
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # Calculate number of parameters for non-attention layers per layer
        mlp_N = hidden_size * intermediate_size * 3  # SwiGLU (gate+up+down)
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2  # Embedding + LM head
        
        # Total non-attention parameters across all layers
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        dense_N_flops = 6 * dense_N * tokens_sum  # 6× (forward+backward+gradient) × tokens

        # Calculate attention layer FLOPS
        seqlen_square_sum = sum(seqlen * seqlen for seqlen in batch_seqlens)
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # Total FLOPS and convert to TFLOPS
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved