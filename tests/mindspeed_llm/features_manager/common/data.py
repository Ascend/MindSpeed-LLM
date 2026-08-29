from mindspeed.features_manager.feature import MindSpeedFeature


class DataFeature(MindSpeedFeature):
    def __init__(self):
        super(DataFeature, self).__init__(feature_name="data", optimization_level=0)

    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--enable-share-memory', action='store_true', default=False,
                            help='Enable shared memory for passing actual_seq_len when reset-position-ids is enabled.')


    def validate_args(self, args):
        if args.enable_share_memory and  args.reset_attention_mask:
            raise AssertionError('Shared memory is not supported  --reset-attention-mask.')
        if args.enable_share_memory and args.position_embedding_type == 'alibi':
            raise AssertionError('Shared memory is not supported with alibi position embeddings.')

    def register_patches(self, patch_manager, args):
        from ...training.utils import get_batch_on_this_tp_rank
        if not args.reset_attention_mask:
            patch_manager.register_patch('megatron.training.utils.get_batch_on_this_tp_rank', 
                                          get_batch_on_this_tp_rank)