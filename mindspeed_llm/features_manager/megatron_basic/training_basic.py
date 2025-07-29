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
        if args.reset_position_ids:
            args.shape_order = 'TND'
            print_rank0_by_args(args, f"When reset_position_ids is enabled, shape_order should be TND.")

        args.create_attention_mask_in_dataloader = False
        reset_data = args.reset_attention_mask
        alibi_without_flash_attn = args.position_embedding_type == 'alibi' and not args.use_flash_attn
        if reset_data or alibi_without_flash_attn or args.tokenizer_padding_side == "left":
            args.create_attention_mask_in_dataloader = True
        if reset_data and args.attention_mask_type == 'causal':
            args.create_attention_mask_in_dataloader = False
        print_rank0_by_args(args, f"[INFO] Setting args.create_attention_mask_in_dataloader to {args.create_attention_mask_in_dataloader} "
                    f"since reset_data={reset_data} or alibi_without_flash_attn={alibi_without_flash_attn} or "
                    f"args.tokenizer_padding_side={args.tokenizer_padding_side}")
        if not args.reset_position_ids and args.neat_pack:
            raise ValueError("Require set `--reset-position-ids` when `--neat-pack` is set.")

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--jit-compile', action='store_true', default=False,
                            help='Setting jit compile mode to True')
        group.add_argument('--attention-mask-type', type=str, default='causal', choices=['causal', 'general'],
                            help='context parallel attention mask type')

    def register_patches(self, patch_manager, args):
        from mindspeed_llm.training.training import train
        from mindspeed_llm.training.checkpointing import load_checkpoint_wrapper
        from mindspeed_llm.tasks.posttrain.lora.utils import is_enable_qlora
        from mindspeed_llm.legacy.data import build_pretraining_data_loader
        from mindspeed_llm.training.utils import get_batch_on_this_tp_rank
        
        patch_manager.register_patch('megatron.training.training.build_pretraining_data_loader',
                                      build_pretraining_data_loader)

        patch_manager.register_patch('megatron.training.utils.get_batch_on_this_tp_rank',
                                      get_batch_on_this_tp_rank)
        

        from mindspeed_llm.training.training import get_model_wrapper
        patch_manager.register_patch('megatron.training.training.get_model',
                                      get_model_wrapper)
        
        patch_manager.register_patch('megatron.training.training.train',
                                      train)
        patch_manager.register_patch('megatron.training.training.load_checkpoint',
                                      load_checkpoint_wrapper)
