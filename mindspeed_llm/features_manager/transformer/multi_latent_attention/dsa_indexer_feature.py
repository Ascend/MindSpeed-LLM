from mindspeed.features_manager.feature import MindSpeedFeature


class DSAIndexerFeature(MindSpeedFeature):
    def __init__(self):
        super(DSAIndexerFeature, self).__init__(feature_name="dsa_indexer", optimization_level=0)

    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)

        group.add_argument('--enable-dsa-indexer', action='store_true', default=False,
                           help='add dsa_indexer module in MLA.')
        group.add_argument('--index-n-heads', type=int, default=64,
                           help='dimension for index head number.')
        group.add_argument('--index-head-dim', type=int, default=128,
                           help='dimension for index head dim.')
        group.add_argument('--index-topk', type=int, default=2048,
                           help='top-k for index head')
        group.add_argument('--scale-fmt', type=str, default=None,
                           help='format for quantization scale.')
        group.add_argument('--indexer-loss-coeff', type=float, default=1.0,
                           help='Indexer loss coeff.')
        group.add_argument('--use-fused-lightning-indexer', action='store_true', default=False,
                           help='Use fused fused operator in lightning indexer.')

    def validate_args(self, args):
        if args.enable_dsa_indexer:
            if not args.multi_latent_attention:
                raise ValueError("DSAIndexer is currently only supported in MLA, plese check model_spec and open --multi-latent-attention.")
            if not args.use_flash_attn:
                raise ValueError("DSAIndexer is currently only supported in FA, plese open --use-flash-attn.")
            if args.context_parallel_size > 1 and args.context_parallel_algo!='ulysses_cp_algo':
                raise ValueError("DSAIndexer is currently only supported `ulysses_cp_algo` when use context parallel.")
            

    def register_patches(self, patch_manager, args):
        if args.enable_dsa_indexer:
            from mindspeed_llm.tasks.models.transformer.dsa_indexer import fp16module_init_wrapper
            patch_manager.register_patch('megatron.core.transformer.module.Float16Module.__init__',
                                         fp16module_init_wrapper)
            from mindspeed_llm.tasks.models.transformer.dsa_indexer import forward_step_dsa_wrapper
            patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.forward_step',
                                         forward_step_dsa_wrapper)
