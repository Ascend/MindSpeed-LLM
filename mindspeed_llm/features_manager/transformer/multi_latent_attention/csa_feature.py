from mindspeed.features_manager.feature import MindSpeedFeature


_CSA_FB_OVERLAP_ATTENTION_FORWARD_PATCHES = (
    'mindspeed.core.transformer.moe.moe_feature.fb_overlap.modules.attention.attention_forward',
    'mindspeed.core.transformer.moe.moe_feature.fb_overlap.overlap_funcs.fwd.attention_forward',
    'mindspeed.core.transformer.moe.moe_feature.fb_overlap.overlap_funcs.fwdbwd.attention_forward',
)

_CSA_BALANCED_MOE_ATTENTION_FORWARD_PATCHES = (
    'mindspeed.core.transformer.moe.moe_feature.balanced_moe.overlap_funcs.fwd.attention_forward',
    'mindspeed.core.transformer.moe.moe_feature.balanced_moe.overlap_funcs.fwdbwd.attention_forward',
)


class CSAFeature(MindSpeedFeature):
    """Register arguments shared by DeepSeek-V4 CSA and HCA."""

    def __init__(self):
        super().__init__('compressed-sparse-attention', optimization_level=0)

    def register_args(self, parser):
        self.add_parser_argument_choices_value(parser, "--position-embedding-type", 'deepseek4')

        group = parser.add_argument_group(title='DeepSeek-V4 CSA/HCA attention')

        # Parameters shared by DeepSeek-V4 Compressed Sparse Attention (CSA)
        # and Heavily Compressed Attention (HCA).
        group.add_argument(
            '--o-groups',
            type=int,
            default=8,
            help='Number of output groups in DeepSeek-V4 CSA/HCA.',
        )
        group.add_argument(
            '--o-lora-rank',
            type=int,
            default=1024,
            help='Output LoRA rank in DeepSeek-V4 CSA/HCA.',
        )
        group.add_argument(
            '--sliding-window-size',
            type=int,
            default=128,
            help='Sliding window size in DeepSeek-V4 CSA/HCA.',
        )
        group.add_argument(
            '--recompute-csa-attention',
            action='store_true',
            default=False,
            help='Enable fine-grained recompute for supported DeepSeek-V4 CSA attention modules.',
        )

    def register_patches(self, patch_manager, args):
        if not (
            getattr(args, 'position_embedding_type', None) == 'deepseek4'
            and getattr(args, 'moe_fb_overlap', False)
            and getattr(args, 'recompute_csa_attention', False)
        ):
            return

        from mindspeed_llm.tasks.models.transformer.deepseek4.csa_fb_overlap import (
            csa_fb_overlap_attention_forward_wrapper,
        )

        patch_targets = list(_CSA_FB_OVERLAP_ATTENTION_FORWARD_PATCHES)
        if getattr(args, 'balanced_moe_experts', False):
            patch_targets.extend(_CSA_BALANCED_MOE_ATTENTION_FORWARD_PATCHES)

        for target in patch_targets:
            patch_manager.register_patch(target, csa_fb_overlap_attention_forward_wrapper)

    def pre_validate_args(self, args):
        # Megatron only allows MTP with rope/none. DeepSeek-V4 supports MTP with its custom
        # position embedding, so temporarily expose it as rope during Megatron validation.
        self.origin_position_embedding_type = None
        if getattr(args, 'mtp_num_layers', None) and getattr(args, 'position_embedding_type', None) == 'deepseek4':
            self.origin_position_embedding_type = args.position_embedding_type
            args.position_embedding_type = 'rope'

    def post_validate_args(self, args):
        # Restore the custom type so model construction still uses DeepSeek-V4 position embedding.
        if self.origin_position_embedding_type is not None:
            args.position_embedding_type = self.origin_position_embedding_type
