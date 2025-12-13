# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import is_megatron_training_available


class ResetAttentionMaskFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('reset-attention-mask', optimization_level=2)

    def register_patches(self, patch_manager, args):
        if getattr(args, self.feature_name, None):
            from mindspeed.core.transformer.flash_attention.reset_attention_mask.utils import (
                _get_ltor_masks_and_position_ids, collate_wrapper, eod_gptdataset_getitem)
            from mindspeed.core.transformer.flash_attention.reset_attention_mask.adaptor import (
                _p2p_ops_eod, rotary_forward, Eod_get_rotary_seq_len)

            patch_manager.register_patch('megatron.core.datasets.gpt_dataset._get_ltor_masks_and_position_ids',
                                         _get_ltor_masks_and_position_ids)
            patch_manager.register_patch('torch.utils.data._utils.collate.default_collate', collate_wrapper)
            patch_manager.register_patch('megatron.core.datasets.gpt_dataset.GPTDataset.__getitem__',
                                         eod_gptdataset_getitem)
            from mindspeed_llm.training.utils import get_batch_on_this_cp_rank_wrapper
            patch_manager.register_patch('megatron.training.utils.get_batch_on_this_cp_rank',
                                         get_batch_on_this_cp_rank_wrapper)
            from mindspeed_llm.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb_thd
            patch_manager.register_patch(
                'megatron.core.models.common.embeddings.rotary_pos_embedding._apply_rotary_pos_emb_thd',
                apply_rotary_pos_emb_thd)
            from mindspeed_llm.core.models.gpt.gpt_model import gpt_forward_wrapper
            patch_manager.register_patch('megatron.core.models.gpt.gpt_model.GPTModel.forward', gpt_forward_wrapper)
            from mindspeed_llm.core.transformer.attention import attention_forward
            patch_manager.register_patch('megatron.core.transformer.attention.Attention.forward', attention_forward)

            patch_manager.register_patch(
                'megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.forward', rotary_forward)
            patch_manager.register_patch(
                'megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.get_rotary_seq_len',
                Eod_get_rotary_seq_len)

            if int(getattr(args, 'context_parallel_size', 1)) > 1:
                patch_manager.register_patch('megatron.core.pipeline_parallel.p2p_communication._p2p_ops', _p2p_ops_eod)
                patch_manager.register_patch('megatron.core.pipeline_parallel.p2p_communication._batched_p2p_ops',
                                             _p2p_ops_eod)
                from mindspeed.core.context_parallel.adaptor import MindSpeedCPDotProductAttention
                patch_manager.register_patch('megatron.core.transformer.dot_product_attention.DotProductAttention',
                                             MindSpeedCPDotProductAttention)
                patch_manager.register_patch('megatron.core.extensions.transformer_engine.TEDotProductAttention',
                                             MindSpeedCPDotProductAttention)

                from mindspeed.core.context_parallel.adaptor import attention_init_wrapper
                patch_manager.register_patch('megatron.core.transformer.attention.Attention.__init__',
                                             attention_init_wrapper)

                from mindspeed.core.context_parallel.model_parallel_utils import initialize_model_parallel_cp_wrapper, \
                    destroy_model_parallel_cp_wrapper, get_context_parallel_group_for_send_recv_overlap

                patch_manager.register_patch('megatron.core.parallel_state.initialize_model_parallel',
                                             initialize_model_parallel_cp_wrapper)
                patch_manager.register_patch('megatron.core.parallel_state.destroy_model_parallel',
                                             destroy_model_parallel_cp_wrapper)
                patch_manager.register_patch(
                    'megatron.core.parallel_state.get_context_parallel_group_for_send_recv_overlap',
                    get_context_parallel_group_for_send_recv_overlap)

                megatron_training_available = is_megatron_training_available()
                if megatron_training_available:
                    from mindspeed.core.context_parallel.get_batch_utils import get_batch_on_this_cp_rank
                    patch_manager.register_patch('megatron.training.utils.get_batch_on_this_cp_rank',
                                                 get_batch_on_this_cp_rank)

                from mindspeed.core.context_parallel.rotary_pos_embedding_utils import get_pos_emb_on_this_cp_rank
                patch_manager.register_patch(
                    'megatron.core.models.common.embeddings.rotary_pos_embedding.get_pos_emb_on_this_cp_rank',
                    get_pos_emb_on_this_cp_rank)

            from mindspeed_llm.training.utils import get_batch_on_this_tp_rank
            patch_manager.register_patch('megatron.training.utils.get_batch_on_this_tp_rank',
                                         get_batch_on_this_tp_rank)

            from mindspeed_llm.core import apply_rotary_pos_emb_bshd
            patch_manager.register_patch('mindspeed.core.fusions.fused_rope.apply_rotary_pos_emb_bshd',
                                         apply_rotary_pos_emb_bshd)

