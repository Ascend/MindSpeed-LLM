from mindspeed.features_manager.feature import MindSpeedFeature


class TransformerBlockFeature(MindSpeedFeature):
    def __init__(self):
        super(TransformerBlockFeature, self).__init__(feature_name="transformer-block", optimization_level=0)
    
    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--first-k-dense-replace', type=int, default=None, 
                            help='Set first k layer as dense layer')

    def validate_args(self, args):
        if args.first_k_dense_replace and args.num_layers <= args.first_k_dense_replace:
            raise AssertionError('Num-layer ({}) must be greater than first-k-dense-replace ({}) when first-k-dense-replace is set.'.format(args.num_layers,
            args.first_k_dense_replace))
        if args.first_k_dense_replace and args.pipeline_model_parallel_size > 1:
            if args.first_k_dense_replace >= args.num_layers // args.pipeline_model_parallel_size:
                raise AssertionError('When using first-k-dense-replace, it is not allowed for all layers within a pp stage to be dense layers.')
        if args.num_experts is not None and args.use_ascend_mc2 and args.moe_grouped_gemm:
            raise AssertionError('Moe Grouped Gemm is not supported with mc2 in MOE model.')

        if args.num_layer_list:
            if len(args.num_layer_list.split(',')) != args.pipeline_model_parallel_size:
                raise ValueError("len(args.num_layer_list) != args.pipeline_model_parallel_size")
            if not args.pipeline_model_parallel_size > 1:
                raise ValueError("Dynamic pipeline model should work with pipeline parallel.")
            if args.num_layers_per_virtual_pipeline_stage:
                raise ValueError("Dynamic pipeline model and virtual pipeline cannot be enabled at the same time.")

        if args.use_ascend_mc2 and args.use_ascend_coc:
            raise AssertionError('--mc2 and coc can not be used together')

    def register_patches(self, patch_manager, args):
        from mindspeed_llm.core.transformer.transformer_block import _transformer_block_build_layers, transformer_block_init_wrapper, transformer_block_forward
        from mindspeed_llm.core.transformer.mlp import core_mlp_init_wrapper

        patch_manager.register_patch('megatron.core.transformer.transformer_block.TransformerBlock._build_layers',
                                      _transformer_block_build_layers)

        # Transformer block
        patch_manager.register_patch('megatron.core.transformer.transformer_block.TransformerBlock.__init__',
                                    transformer_block_init_wrapper)
        patch_manager.register_patch('megatron.core.transformer.transformer_block.TransformerBlock.forward',
                                    transformer_block_forward)

        patch_manager.register_patch('megatron.core.transformer.mlp.MLP.__init__', core_mlp_init_wrapper)

        if args.share_kvstates:
            from mindspeed_llm.core.transformer.transformer_block import share_kvstates_checkpointed_forward_func
            patch_manager.register_patch('megatron.core.transformer.transformer_block.TransformerBlock._checkpointed_forward',
                                        share_kvstates_checkpointed_forward_func)