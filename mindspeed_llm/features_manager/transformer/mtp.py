from mindspeed.features_manager.feature import MindSpeedFeature


class MultiTokenPredictionFeature(MindSpeedFeature):
    def __init__(self):
        super(MultiTokenPredictionFeature, self).__init__(feature_name="multi-token-prediction", optimization_level=0)
    
    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)

        group.add_argument('--mtp-mem-efficient-logits', action='store_true', default=False,
                            help='Optimize ce_loss memory when use mtp block.')
        group.add_argument('--mtp-after-norm', action='store_true', default=False,
                            help='Optimize ce_loss memory when use mtp block.')

    def register_patches(self, patch_manager, args):
        import megatron
        from mindspeed_llm.core import PTNorm
        from mindspeed_llm.core.transformer.multi_token_prediction import get_mtp_num_layers_to_build, \
            mtp_reduce_loss_in_tracker

        # dualpipe do not need to init embedding weight
        # Use existing patch: megatron.core.models.common.embeddings.language_model_embedding.LanguageModelEmbedding.__init__
        # mtp compatibility
        megatron.core.transformer.multi_token_prediction.LNImpl = PTNorm
        patch_manager.register_patch('megatron.core.transformer.multi_token_prediction.MTPLossLoggingHelper.reduce_loss_in_tracker',
                                      mtp_reduce_loss_in_tracker)
        
        # mtp memory optimization
        if args.mtp_mem_efficient_logits:
            from mindspeed_llm.core.tensor_parallel.cross_entropy import calculate_logits_max
            patch_manager.register_patch('megatron.core.tensor_parallel.cross_entropy.VocabParallelCrossEntropy.calculate_logits_max',
                                          calculate_logits_max)

        # patch for mtp
        from mindspeed_llm.core.transformer.multi_token_prediction import (
            mtp_layer_init_wrapper,
            mtp_layer_forward,
            mtp_block_build_layers_wrapper,
            mtp_block_forward,
        )
        patch_manager.register_patch('megatron.core.transformer.multi_token_prediction.MultiTokenPredictionLayer.__init__',
                                      mtp_layer_init_wrapper)
        if not args.schedules_method == "dualpipev":
            patch_manager.register_patch('megatron.core.transformer.multi_token_prediction.MultiTokenPredictionLayer.forward',
                                          mtp_layer_forward, force_patch=True)
        patch_manager.register_patch('megatron.core.transformer.multi_token_prediction.MultiTokenPredictionBlock._build_layers',
                                      mtp_block_build_layers_wrapper)
        patch_manager.register_patch('megatron.core.transformer.multi_token_prediction.MultiTokenPredictionBlock.forward',
                                      mtp_block_forward)