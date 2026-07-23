from mindspeed.features_manager.feature import MindSpeedFeature


class DSAIndexerFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__(feature_name="dsa_indexer", optimization_level=0)

    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)

        group.add_argument(
            '--enable-dsa-indexer', action='store_true', default=False, help='add dsa_indexer module in MLA.'
        )
        group.add_argument(
            '--init-norm-weight-in-fp32',
            action='store_true',
            default=False,
            help='initialize weights of the normalization layer in fp32 format.',
        )
        group.add_argument('--index-n-heads', type=int, default=64, help='dimension for index head number.')
        group.add_argument('--index-head-dim', type=int, default=128, help='dimension for index head dim.')
        group.add_argument('--index-topk', type=int, default=2048, help='top-k for index head')
        group.add_argument('--scale-fmt', type=str, default=None, help='format for quantization scale.')
        group.add_argument('--indexer-loss-coeff', type=float, default=1.0, help='Indexer loss coeff.')
        group.add_argument(
            '--use-fused-lightning-indexer',
            action='store_true',
            default=False,
            help='Use fused fused operator in lightning indexer.',
        )
        group.add_argument(
            '--use-fused-lightning-indexer-loss',
            action='store_true',
            default=False,
            help='Use fused fused operator in lightning indexer.',
        )

        # compress arguments
        group.add_argument(
            '--kv-compress', action='store_true', default=False, help='Apply compress to kv computations.'
        )
        group.add_argument('--compress-ratios', type=int, nargs='+', default=None, help='Compress ratios of layers.')
        group.add_argument('--rope-head-dim', type=int, default=64, help='rope head dim.')
        group.add_argument('--norm-eps', type=float, default=1e-6, help='norm-eps.')
        group.add_argument('--max-batch-size', type=int, default=4, help='rope head dim.')
        group.add_argument('--original-seq-len', type=int, default=65536, help='')
        group.add_argument('--compress-rope-theta', type=float, default=40000.0, help='')
        group.add_argument('--rope-theta', type=float, default=10000.0, help='')
        group.add_argument('--rope-factor', type=float, default=4.0, help='')

        group.add_argument(
            '--index-topk-freq',
            type=int,
            default=1,
            help='Frequency of topk computation for IndexCache. '
            'Layers with (layer_id - index_skip_topk_offset + 1) %% index_topk_freq != 0 '
            'will skip topk computation and reuse cached indices.',
        )
        group.add_argument(
            '--index-topk-pattern',
            type=str,
            default=None,
            help='Pattern string for which layers skip topk (S=skip, C=compute). '
            'If provided, overrides index_topk_freq logic.',
        )
        group.add_argument(
            '--index-skip-topk-offset',
            type=int,
            default=2,
            help='Offset for skip topk computation in IndexCache. '
            'Layers before this offset always compute their own indices.',
        )
        group.add_argument(
            '--index-share-for-mtp-iteration',
            action='store_true',
            default=False,
            help='Share topk indices across MTP iterations. '
            'When enabled, step 0 computes indices and steps 1+ reuse them.',
        )
        group.add_argument(
            '--apply-rope-no-in-complex',
            action='store_true',
            default=False,
            help='No Use complex number implementation of RoPE, only works with --enable-dsa-indexer.',
        )
        group.add_argument(
            '--no-use-sparse-c8-indexer',
            action='store_true',
            default=False,
            help='No Use c8 indexer.',
        )
        group.add_argument(
            '--indexer-qk-quant-scheme',
            type=str,
            default=None,
            choices=['mxfp4', 'mxfp8'],
            help='Support quant type for Indexer. '
            '(Effective only when --enable-dsa-indexer is set, '
            'requires --kv-compress, and is incompatible with --use-fused-lightning-indexer.)',
        )

    def validate_args(self, args):
        if args.enable_dsa_indexer:
            if not args.multi_latent_attention:
                raise ValueError(
                    "DSAIndexer is currently only supported in MLA, plese check model_spec and open --multi-latent-attention."
                )
            if not args.use_flash_attn:
                raise ValueError("DSAIndexer is currently only supported in FA, plese open --use-flash-attn.")
            if args.context_parallel_size > 1 and args.context_parallel_algo not in [
                'ulysses_cp_algo',
                'kvallgather_cp_algo',
            ]:
                raise ValueError("DSAIndexer is currently only supported `ulysses_cp_algo` when use context parallel.")
            if args.reset_attention_mask:
                if not args.use_fused_lightning_indexer:
                    raise ValueError("DSA with TND format requires --use-fused-lightning-indexer.")
                if not args.use_fused_lightning_indexer_loss:
                    raise ValueError("DSA with TND format requires --use-fused-lightning-indexer-loss.")
                if not args.use_sparse_flash_attn:
                    raise ValueError("DSA with TND format requires --use-sparse-flash-attn.")
        if args.indexer_qk_quant_scheme is not None:
            if not args.enable_dsa_indexer:
                raise ValueError(
                    "--index-quant-scheme requires --enable-dsa-indexer to be enabled. "
                    "Please set --enable-dsa-indexer or remove --index-quant-scheme."
                )
            if not args.kv_compress:
                raise ValueError(
                    "--index-quant-scheme requires --kv-compress to be enabled. "
                    "Please set --kv-compress or remove --index-quant-scheme."
                )
            if args.use_fused_lightning_indexer:
                raise ValueError(
                    "--index-quant-scheme is not compatible with --use-fused-lightning-indexer. "
                    "Please disable --use-fused-lightning-indexer or remove --index-quant-scheme."
                )

    def register_patches(self, patch_manager, args):
        if args.enable_dsa_indexer:
            from mindspeed_llm.tasks.models.transformer.dsa_indexer import forward_step_dsa_wrapper

            patch_manager.register_patch(
                'megatron.core.pipeline_parallel.schedules.forward_step', forward_step_dsa_wrapper
            )
            if args.moe_fb_overlap:
                patch_manager.register_patch(
                    'mindspeed.core.transformer.moe.moe_feature.fb_overlap.vpp_schedules.forward_step',
                    forward_step_dsa_wrapper,
                )

        if args.init_norm_weight_in_fp32:
            from mindspeed_llm.tasks.models.transformer.dsa_indexer import norm2fp32_fp16module_init_wrapper

            patch_manager.register_patch(
                'megatron.core.transformer.module.Float16Module.__init__', norm2fp32_fp16module_init_wrapper
            )
