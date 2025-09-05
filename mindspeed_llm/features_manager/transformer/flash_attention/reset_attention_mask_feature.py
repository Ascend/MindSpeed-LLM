# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class ResetAttentionMaskFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('reset-attention-mask', optimization_level=2)

    def register_patches(self, patch_manager, args):
        if getattr(args, self.feature_name, None) and int(args.context_parallel_size) > 1 and (
                args.context_parallel_algo in ['megatron_cp_algo', 'hybrid_cp_algo']):
            from mindspeed.core.transformer.flash_attention.reset_attention_mask.utils import (
                    _get_ltor_masks_and_position_ids, collate_wrapper, get_batch_on_this_cp_rank_wrapper)
            from mindspeed.core.transformer.flash_attention.reset_attention_mask.adaptor import (
                    _p2p_ops_eod, gpt_forward_wrapper, attention_forward, apply_rotary_pos_emb_thd,
                    rotary_forward, Eod_get_rotary_seq_len)
            from mindspeed.core.context_parallel.get_batch_utils import get_batch_on_this_tp_rank
        
            patch_manager.register_patch(
                    'megatron.core.models.common.embeddings.rotary_pos_embedding._apply_rotary_pos_emb_thd',
                    apply_rotary_pos_emb_thd)
        
            from mindspeed_llm.core.models.gpt.gpt_model import gpt_forward_wrapper
            patch_manager.register_patch('megatron.core.models.gpt.gpt_model.GPTModel.forward', gpt_forward_wrapper) 
        
            from mindspeed.core.tensor_parallel.unaligned_layers.adaptor import  get_rotary_seq_len
            patch_manager.register_patch(
                    'megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.get_rotary_seq_len',
                    get_rotary_seq_len)
        
            patch_manager.register_patch('megatron.core.transformer.attention.Attention.forward', attention_forward)

            from mindspeed.core.context_parallel.adaptor import MindSpeedCPDotProductAttention
            patch_manager.register_patch('megatron.core.transformer.dot_product_attention.DotProductAttention', 
                    MindSpeedCPDotProductAttention)
            patch_manager.register_patch('megatron.core.extensions.transformer_engine.TEDotProductAttention',
                    MindSpeedCPDotProductAttention)

            from mindspeed.core.context_parallel.model_parallel_utils import initialize_model_parallel_cp_wrapper
            patch_manager.register_patch('megatron.core.parallel_state.initialize_model_parallel',
                    initialize_model_parallel_cp_wrapper)
