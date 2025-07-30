from mindspeed.features_manager.feature import MindSpeedFeature


class MoERouter(MindSpeedFeature):
    def __init__(self):
        super(MoERouter, self).__init__(feature_name="moe_router", optimization_level=0)
        

    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)
        self.add_parser_argument_choices_value(parser, "--moe-router-load-balancing-type", 'group_limited_greedy')
        self.add_parser_argument_choices_value(parser, "--moe-router-load-balancing-type", 'softmax_topk')
        self.add_parser_argument_choices_value(parser, "--moe-router-load-balancing-type", 'pai_megatron_aux_loss')
        self.add_parser_argument_choices_value(parser, "--moe-router-load-balancing-type", 'sparsemixer_topk')
        self.add_parser_argument_choices_value(parser, "--moe-router-load-balancing-type", 'noaux_tc')                                
        group.add_argument('--topk-group', type=int, default=None,
                            help='Choose topK group experts in group_limited_greedy_topK method')
        group.add_argument('--routed-scaling-factor', type=float, default=None,
                            help='The routed scaling factor')
        group.add_argument('--norm-topk-prob', action='store_true', default=False, 
                            help='Normalize the topk weight')
        group.add_argument('--n-group', type=int, default=None,
                            help='Number of groups for routed experts.'
                                 'Tips: in deepseek3, set n-group equal to EP to limit each token to experts on a subset of devices,'
                                 'set n-group equal to number of nodes in EP group to limit each token to experts on a subset of nodes.')
        group.add_argument('--seq-aux', action='store_true', default=False, 
                            help='Compute aux loss in seq_aux')
        group.add_argument('--moe-device-level-aux-loss-coeff', type=float, default=0.,
                            help='set the coeff for devicie-level balance loss in deepseek moe')
        group.add_argument('--moe-comm-aux-loss-coeff', type=float, default=0.,
                            help='set the coeff for communication balance loss in deepseek moe')
        group.add_argument('--router-gating-in-fp32', action='store_true', default=False,
                            help='Compute router gating in float32.')
        group.add_argument("--moe-revert-type-after-topk", action='store_true',
                            help="revert the type of logits after the topk has been computed")
        group.add_argument("--fix-router", action='store_true', 
                            help="fix router for load balancing.")

    def pre_validate_args(self, args):
        self.origin_spec = None
        self.origin_spec = args.spec
        args.spec = None

    def validate_args(self, args):
        self._validate_moe_args(args)
        self._validate_group_limited_greedy(args)
        self._validate_aux_loss_free(args)

    def post_validate_args(self, args):
        if self.origin_spec:
            args.spec = self.origin_spec

    def _validate_moe_args(self, args):
        from mindspeed_llm.training.utils import print_rank0_by_args

        if args.moe_expert_capacity_factor is not None:
            if args.moe_token_dispatcher_type == "allgather":
                raise ValueError(f'moe_expert_capacity_factor not works with allgather token dispatcher')
            if args.moe_expert_capacity_factor < 0:
                args.moe_expert_capacity_factor = None
                print_rank0_by_args(
                    f'When moe_expert_capacity_factor < 0, no token would be drop, so moe_expert_capacity_factor should be set to false.')
            if args.moe_router_load_balancing_type not in ["aux_loss", "none"]:
                raise ValueError(f'moe_expert_capacity_factor only works with aux_loss or none load balancing')
            if args.moe_expert_capacity_factor is None and args.moe_pad_expert_input_to_capacity:
                raise ValueError(f'moe_expert_capacity_factor must be set to use moe_pad_expert_input_to_capacity')
            if args.shared_expert_gate_output_dimension != 1 and args.shared_expert_gate_output_dimension != args.hidden_size:
                raise AssertionError('shared expert gate output dimension can only be configured with 1 or hidden_size')
            if hasattr(args,
                       'use_fused_moe_token_permute_and_unpermute') and args.use_fused_moe_token_permute_and_unpermute:
                raise AssertionError(
                    'moe_expert_capacity_factor mode does not support use_fused_moe_token_permute_and_unpermute')

    def _validate_group_limited_greedy(self, args):
        if args.moe_router_load_balancing_type == "group_limited_greedy":
            if args.topk_group is None:
                raise AssertionError('The parameter topk-group should be set when use group_limited_greedy.')
            elif args.routed_scaling_factor is None:
                raise AssertionError(
                    'The parameter routed_scaling_factor should be set when use multi_latent_attention.')
            elif args.topk_group >= args.expert_model_parallel_size:
                raise AssertionError('The topk group ({}) should be less than n-group(EP)({}).'.format(args.topk_group,
                                                                                                       args.expert_model_parallel_size))

    def _validate_aux_loss_free(self, args):
        if args.moe_router_enable_expert_bias and args.moe_router_score_function != "sigmoid":
            raise ValueError(
                "Expert bias for aux-loss-free routing only supports sigmoid score function."
                "Please set --moe-router-score-function sigmoid for sigmoid score function."
            )

    def register_patches(self, patch_manager, args):
        from ...core import (topk_router_forward, topk_router_routing, z_loss_func, topk_router_init_wrapper,
                             topk_router_gating_func)

        patch_manager.register_patch('megatron.core.transformer.moe.router.TopKRouter.__init__', 
                                      topk_router_init_wrapper)
        patch_manager.register_patch('megatron.core.transformer.moe.router.TopKRouter.routing', 
                                      topk_router_routing)
        patch_manager.register_patch('megatron.core.transformer.moe.router.TopKRouter.forward', 
                                      topk_router_forward)
        patch_manager.register_patch('megatron.core.transformer.moe.router.TopKRouter.gating', 
                                      topk_router_gating_func)
        patch_manager.register_patch('megatron.core.transformer.moe.router.z_loss_func', 
                                      z_loss_func)
