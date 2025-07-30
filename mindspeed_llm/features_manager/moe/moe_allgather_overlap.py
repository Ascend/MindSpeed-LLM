from mindspeed.features_manager.moe.moe_allgather_overlap import MoEAllGatherOverLapFeature as MindSpeedMoEAllGatherOverLapFeature


class MoEAllGatherOverLapFeature(MindSpeedMoEAllGatherOverLapFeature):
            
    def register_patches(self, patch_manager, args):
        from mindspeed.core.transformer.moe.moe_feature.adaptor import MindSpeedAllGatherOverlapMoeLayerAdaptor
        from mindspeed.core.transformer.moe.moe_feature.overlap.moe_common import mlp_init, core_mlp_forward_wrapper
        from mindspeed_llm.core.transformer.moe.moe_layer import parallel_transformer_layer_init_wrapper

        patch_manager.register_patch('megatron.core.transformer.mlp.MLP.forward',
                                      core_mlp_forward_wrapper)

        if getattr(args, 'moe_token_dispatcher_type', None) == "allgather":
            if args.moe_allgather_overlap_comm:
                patch_manager.register_patch('megatron.core.transformer.moe.moe_layer.MoELayer',
                                              MindSpeedAllGatherOverlapMoeLayerAdaptor)
                patch_manager.register_patch('megatron.core.transformer.mlp.MLP.__init__',
                                              mlp_init)
                patch_manager.register_patch('megatron.core.transformer.transformer_layer.TransformerLayer.__init__',
                                              parallel_transformer_layer_init_wrapper)  