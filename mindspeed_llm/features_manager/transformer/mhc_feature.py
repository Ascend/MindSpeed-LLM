from mindspeed.features_manager.feature import MindSpeedFeature


class MHCFeature(MindSpeedFeature):
    def __init__(self):
        super(MHCFeature, self).__init__(feature_name="mhc", optimization_level=0)

    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)

        group.add_argument('--enable-mhc', action='store_true', default=False,
                           help='add mhc module in model.')
        group.add_argument('--use-triton-rmsnorm-without-weight', action='store_true', default=False,
                           help='use triton rmsnorm-without-weight.')
        group.add_argument('--use-triton-sinkhorn', action='store_true', default=False,
                           help='use triton sinkhorn.')
        group.add_argument('--hc-mult', type=int, default=4,
                           help='dimension for index head number.')
        group.add_argument('--hc-sinkhorn-iters', type=int, default=20,
                           help='dimension for index head dim.')
        group.add_argument('--hc-eps', type=float, default=1e-6,
                           help='dimension for index head dim.')
        group.add_argument('--use-triton-mhc', action='store_true', default=False,
                           help='use triton for pre/pos/only pre.')
        group.add_argument('--mhc-recompute', action='store_true', default=False,
                           help='Fine-grained recompute for attention pre/pos and mlp pre.')

    def register_patches(self, patch_manager, args):
        if args.enable_mhc:
            # adapt mhc in PP stage
            from mindspeed_llm.tasks.models.transformer.deepseek4.mhc.mhc import get_tensor_shapes_in_mhc
            patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.get_tensor_shapes', get_tensor_shapes_in_mhc)
            
            if getattr(args, "num_layers_per_virtual_pipeline_stage", False) and args.num_layers_per_virtual_pipeline_stage is not None:
                from mindspeed_llm.tasks.models.transformer.deepseek4.mhc.mhc import forward_backward_pipelining_with_interleaving_in_mhc
                patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_with_interleaving', forward_backward_pipelining_with_interleaving_in_mhc)

    def validate_args(self, args):
        if args.enable_mhc:
            if not args.multi_latent_attention:
                raise ValueError("DSAIndexer is currently only supported in MLA, plese check model_spec and open --multi-latent-attention.")
            if not args.use_flash_attn:
                raise ValueError("DSAIndexer is currently only supported in FA, plese open --use-flash-attn.")
            valid_algos = ['ulysses_cp_algo', 'kvallgather_cp_algo']
            if args.context_parallel_size > 1 and args.context_parallel_algo not in valid_algos:
                raise ValueError("DSAIndexer is currently only supported `ulysses_cp_algo` when use context parallel.")
