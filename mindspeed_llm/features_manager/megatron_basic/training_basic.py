import os
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


def print_rank0_by_args(args, message):
    """Before initialization of distributed, we only print on rank 0."""
    if args.rank == 0:
        print(message, flush=True)


class TrainingBasicFeature(MindSpeedFeature):
    
    def __init__(self):
        super(TrainingBasicFeature, self).__init__(feature_name="training", optimization_level=0)

    def pre_validate_args(self, args):
        args.use_mcore_models = not args.use_legacy_models
        if args.reset_attention_mask:
            args.shape_order = 'TND'
            print_rank0_by_args(args, f"When reset_attention_mask is enabled, shape_order should be TND.")

        args.create_attention_mask_in_dataloader = False
        alibi_without_flash_attn = args.position_embedding_type == 'alibi' and not args.use_flash_attn
        if (args.reset_attention_mask and args.attention_mask_type == 'general') or alibi_without_flash_attn or args.tokenizer_padding_side == "left":
            args.create_attention_mask_in_dataloader = True
        if args.reset_attention_mask and args.attention_mask_type == 'causal':
            args.create_attention_mask_in_dataloader = False
        # Temporary code modification
        if args.attention_mask_type == 'general' and args.context_parallel_algo == 'ulysses_cp_algo' and args.reset_attention_mask:
            args.create_attention_mask_in_dataloader = False

        print_rank0_by_args(args, f"[INFO] Setting args.create_attention_mask_in_dataloader to {args.create_attention_mask_in_dataloader} "
                    f"since reset_attention_mask={args.reset_attention_mask} or alibi_without_flash_attn={alibi_without_flash_attn} or "
                    f"args.tokenizer_padding_side={args.tokenizer_padding_side}")
        if not args.reset_attention_mask and args.neat_pack:
            raise ValueError("Require set `--reset-attention-mask` when `--neat-pack` is set.")

        # Bypass megatron validation when pp == 2 and vpp is enabled.
        self.origin_num_layers_per_virtual_pipeline_stage = None
        self.origin_noverlap_p2p_comm = None
        if args.pipeline_model_parallel_size == 2 and args.num_layers_per_virtual_pipeline_stage is not None:
            self.origin_num_layers_per_virtual_pipeline_stage = args.num_layers_per_virtual_pipeline_stage
            self.origin_noverlap_p2p_comm = args.overlap_p2p_comm
            args.num_layers_per_virtual_pipeline_stage = None
            args.overlap_p2p_comm = None

    def validate_args(self, args):
        # mitigate FSDP2 performance degradation
        if getattr(args, "use_torch_fsdp2", False):
            os.environ['MULTI_STREAM_MEMORY_REUSE'] = '2'

    def post_validate_args(self, args):
        if self.origin_num_layers_per_virtual_pipeline_stage:
            args.num_layers_per_virtual_pipeline_stage = self.origin_num_layers_per_virtual_pipeline_stage
            args.overlap_p2p_comm = self.origin_noverlap_p2p_comm

        """validate scenario that vpp is enabled when pp=2."""
        if args.pipeline_model_parallel_size != 2 or args.num_layers_per_virtual_pipeline_stage is None:
            return

        if args.num_layers_per_virtual_pipeline_stage and args.num_layer_list:
            raise AssertionError(
                'num_layers_per_virtual_pipeline_stage is not support work with num_layer_list')

        # VPP enabled when pp == 2, do check.
        num_layers_per_pipeline_stage = args.num_layers // args.pipeline_model_parallel_size
        if num_layers_per_pipeline_stage % args.num_layers_per_virtual_pipeline_stage != 0:
            raise AssertionError('number of layers per pipeline stage must be divisible number of layers per virtual pipeline stage')

        pp_stage_layers = args.num_layers / args.pipeline_model_parallel_size
        if args.num_layers_per_virtual_pipeline_stage and args.num_layers_per_virtual_pipeline_stage >= pp_stage_layers:
            raise ValueError("Num of layers in vpp stage should be less than pp stage, "
                             "please turn down args.num_layers_per_virtual_pipeline_stage.")

        args.virtual_pipeline_model_parallel_size = num_layers_per_pipeline_stage // \
                                                    args.num_layers_per_virtual_pipeline_stage

        print_rank0_by_args(args, f'vpp_size would be {args.virtual_pipeline_model_parallel_size} since '
                                  f'num_layers_per_virtual_pipeline_stage is {args.num_layers_per_virtual_pipeline_stage}')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--jit-compile', action='store_true', default=False,
                            help='Setting jit compile mode to True')
        group.add_argument('--load-checkpoint-loosely', action='store_true', default=False,
                            help='Enable loading checkpoint not strictly.')

    def register_patches(self, patch_manager, args):
        from mindspeed_llm.training.training import train
        from mindspeed_llm.training.checkpointing import load_checkpoint_wrapper
        from mindspeed_llm.tasks.posttrain.lora.utils import is_enable_qlora
        from mindspeed_llm.legacy.data import build_pretraining_data_loader
        from mindspeed_llm.training.utils import get_batch_on_this_tp_rank
        
        patch_manager.register_patch('megatron.training.training.build_pretraining_data_loader',
                                      build_pretraining_data_loader)
        if not getattr(args, 'reset_attention_mask', None):
            patch_manager.register_patch('megatron.training.utils.get_batch_on_this_tp_rank',
                                         get_batch_on_this_tp_rank)

        patch_manager.register_patch('megatron.training.training.train',
                                      train)
        patch_manager.register_patch('megatron.training.training.load_checkpoint',
                                      load_checkpoint_wrapper)