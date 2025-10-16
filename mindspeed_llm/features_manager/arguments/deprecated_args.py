# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import warnings
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class DeprecatedArgsFeature(MindSpeedFeature):

    def __init__(self):
        super(DeprecatedArgsFeature, self).__init__(feature_name="deprecated-args", optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--use-deter-comp', action='store_true', default=False, dest='deprecated_use_deter_comp',
                           help='enable deterministic computing for npu'
                                'Note: this option is deprecated, please use --npu-deterministic instead!')
        group.add_argument('--use-mc2', action='store_true', dest='deprecated_use_mc2',
                           help='enable mc2'
                                'Note: this option is deprecated, please use --use-ascend-mc2 instead!')
        group.add_argument('--cp-attention-mask-type', type=str, default=None, choices=['causal', 'general'], dest='deprecated_cp_attention_mask_type',
                           help='context parallel attention mask type'
                                'Note: this option is deprecated, please use --attention-mask-type instead!')
        group.add_argument('--moe-intermediate-size', type=int, default=None, dest='deprecated_moe_intermediate_size',
                           help='The ffn hidden size of MoE layer'
                                'Note: this option is deprecated, please use --moe-ffn-hidden-size instead!')
        group.add_argument('--n-group', type=int, default=None, dest='deprecated_n_group',
                           help='Number of groups for routed experts.'
                                'Tips: in deepseek3, set n-group equal to EP to limit each token to experts on a subset of devices,'
                                'set n-group equal to number of nodes in EP group to limit each token to experts on a subset of nodes.'
                                'Note: this option is deprecated, please use --moe-router-num-groups instead!')
        group.add_argument('--topk-group', type=int, default=None, dest='deprecated_topk_group',
                           help='Choose topK group experts in group_limited_greedy_topK method,'
                                'Note: this option is deprecated, please use --moe-router-group-topk instead!')
        group.add_argument('--routed-scaling-factor', type=float, default=None, dest='deprecated_routed_scaling_factor',
                           help='The routed scaling factor.'
                                'Note: this option is deprecated, please use --moe-router-topk-scaling-factor instead!')
        group.add_argument('--qk-rope-head-dim', type=int, default=None, dest='deprecated_qk_rope_head_dim',
                           help='The qk head dim for rope'
                                'Note: this option is deprecated, please use --qk-pos-emb-head-dim instead!')
        group.add_argument('--qk-nope-head-dim', type=int, default=None, dest='deprecated_qk_nope_head_dim',
                           help='The qk head dim for only self-attn'
                                'Note: this option is deprecated, please use --qk-head-dim instead!')
        group.add_argument('--multi-head-latent-attention', action='store_true', default=False, dest='deprecated_multi_head_latent_attention',
                           help='Use Multi-head Latent Attention(MLA)'
                                'Note: this option is deprecated, please use --multi-latent-attention instead!')
        group.add_argument('--rope-scaling-beta-fast', type=int, default=None, dest='deprecated_rope_scaling_beta_fast',
                           help='Yarn rope: rope beta fast'
                                'Note: this option is deprecated, please use --beta-fast instead!')
        group.add_argument('--rope-scaling-beta-slow', type=int, default=None, dest='deprecated_rope_scaling_beta_slow',
                           help='Yarn rope: rope beta slow'
                                'Note: this option is deprecated, please use --beta-slow instead!')

    def validate_args(self, args):
        # If deprecated argument are used instead of new argument, we assign the deprecated argument to the new argument and issue a warning
        if args.deprecated_use_deter_comp and not args.npu_deterministic:
            warnings.warn(
                """The '--use-deter-comp' argument is deprecated and will be removed in the next future version, 
                   please use '--npu-deterministic' instead!""", DeprecationWarning)
            args.npu_deterministic = args.deprecated_use_deter_comp
        if args.deprecated_use_mc2 and not args.use_ascend_mc2:
            warnings.warn(
                """The '--use-mc2' argument is deprecated and will be removed in the next future version, 
                   please use '--use-ascend-mc2' instead!""", DeprecationWarning)
            args.use_ascend_mc2 = args.deprecated_use_mc2
        if args.deprecated_cp_attention_mask_type:
            warnings.warn(
                """The '--cp-attention-mask-type' argument is deprecated and will be removed in the next future version, 
                   please use '--attention-mask-type' instead!""", DeprecationWarning)
            args.attention_mask_type = args.deprecated_cp_attention_mask_type
        if args.deprecated_moe_intermediate_size:
            warnings.warn(
                """The '--moe-intermediate-size' argument is deprecated and will be removed in the next future version, 
                   please use '--moe-ffn-hidden-size' instead!""", DeprecationWarning)
            args.moe_ffn_hidden_size = args.deprecated_moe_intermediate_size
        if args.deprecated_n_group and not args.moe_router_num_groups:
            warnings.warn(
                """The '--n-group' argument is deprecated and will be removed in the next future version, 
                   please use '--moe-router-num-groups' instead!""", DeprecationWarning)
            args.moe_router_num_groups = args.deprecated_n_group
        if args.deprecated_topk_group and not args.moe_router_group_topk:
            warnings.warn(
                """The '--topk-group' argument is deprecated and will be removed in the next future version, 
                   please use '--moe-router-group-topk' instead!""", DeprecationWarning)
            args.moe_router_group_topk = args.deprecated_topk_group
        if args.deprecated_routed_scaling_factor and not args.moe_router_topk_scaling_factor:
            warnings.warn(
                """The '--routed-scaling-factor' argument is deprecated and will be removed in the next future version, 
                   please use '--moe-router-topk-scaling-factor' instead!""", DeprecationWarning)
            args.moe_router_topk_scaling_factor = args.deprecated_topk_group
        if args.deprecated_qk_rope_head_dim:
            warnings.warn(
                """The '--qk-rope-head-dim' argument is deprecated and will be removed in the next future version, 
                   please use '--qk-pos-emb-head-dim' instead!""", DeprecationWarning)
            args.qk_pos_emb_head_dim = args.deprecated_qk_rope_head_dim
        if args.deprecated_qk_nope_head_dim:
            warnings.warn(
                """The '--qk-nope-head-dim' argument is deprecated and will be removed in the next future version, 
                   please use '--qk-head-dim' instead!""", DeprecationWarning)
            args.qk_head_dim = args.deprecated_qk_nope_head_dim
        if args.deprecated_multi_head_latent_attention and not args.multi_latent_attention:
            warnings.warn(
                """The '--multi-head-latent-attention' argument is deprecated and will be removed in the next future version, 
                   please use '--multi-latent-attention' instead!""", DeprecationWarning)
            args.multi_latent_attention = args.deprecated_multi_head_latent_attention
        if args.deprecated_rope_scaling_beta_fast:
            warnings.warn(
                """The '--rope-scaling-beta-fast' argument is deprecated and will be removed in the next future version, 
                   please use 'beta-fast' instead!""", DeprecationWarning)
            args.beta_fast = args.deprecated_rope_scaling_beta_fast
        if args.deprecated_rope_scaling_beta_slow:
            warnings.warn(
                """The '--rope-scaling-beta-slow' argument is deprecated and will be removed in the next future version, 
                   please use 'beta-slow' instead!""", DeprecationWarning)
            args.beta_slow = args.deprecated_rope_scaling_beta_slow