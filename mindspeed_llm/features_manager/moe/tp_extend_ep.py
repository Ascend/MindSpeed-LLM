from mindspeed.features_manager.moe.tp_extend_ep import MoETpExtendEpFeature as MindSpeedMoETpExtendEpFeature


class MoETpExtendEpFeature(MindSpeedMoETpExtendEpFeature):

    def register_patches(self, patch_manager, args):
        from mindspeed.core.transformer.moe.moe_feature.adaptor import MindSpeedAlltoAllSEQTptoEpMoELayer

        if hasattr(args, 'moe_token_dispatcher_type') and args.moe_token_dispatcher_type == 'alltoall_seq':
            if args.moe_tp_extend_ep:
                if not args.moe_alltoall_overlap_comm:
                    patch_manager.register_patch('megatron.core.transformer.moe.moe_layer.MoELayer',
                                                  MindSpeedAlltoAllSEQTptoEpMoELayer)
