from mindspeed.features_manager.feature import MindSpeedFeature


class Qwen3NextFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('qwen3-next-attention', optimization_level=0)

    def register_args(self, parser):
        group = parser.add_argument_group(title='qwen3 next attention')

        group.add_argument('--full-attention-interval', type=int, default=0,
                            help='full attention interval')
        group.add_argument('--linear-key-head-dim', type=int, default=0,
                            help='linear key head-dim')
        group.add_argument('--linear-num-key-heads', type=int, default=0,
                            help='linear num key heads')
        group.add_argument('--linear-num-value-heads', type=int, default=0,
                            help='linear num value heads')
        group.add_argument('--linear-value-head-dim', type=int, default=0,
                            help='linear value head dim')
        group.add_argument('--partial-rotary-factor', type=float, default=0.0,
                            help='partial rotary factor')
        group.add_argument('--use-triton-gdn', action="store_true", default=False,
                           help='use triton gdn')
        # for global aux loss
        group.add_argument('--use-global-aux-loss', action='store_true', default=False,
                       help='Use global aux loss for loss calculation.')

    def register_patches(self, patch_manager, args):
        from mindspeed_llm.core.transformer.attention import self_attention_init 
        from mindspeed_llm.core.pipeline_parallel.schedules import global_aux_loss_forward_step
        from mindspeed_llm.core.transformer.moe.router import global_aux_loss_topk_router_forward, global_aux_loss_load_balancing

        patch_manager.register_patch('megatron.core.transformer.attention.SelfAttention.__init__', self_attention_init)   

        if args.use_global_aux_loss: 
            if int(args.pipeline_model_parallel_size) > 1:
                raise AssertionError('use-global-aux-loss is not support for pipeline-model-parallel-size > 1, please use FSDP2')
                
            patch_manager.register_patch('megatron.core.transformer.moe.router.TopKRouter.forward', 
                                        global_aux_loss_topk_router_forward)
            patch_manager.register_patch('megatron.core.transformer.moe.router.TopKRouter.aux_loss_load_balancing', 
                                        global_aux_loss_load_balancing)
            patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.forward_step', global_aux_loss_forward_step)                                                                                                                                                                                                     