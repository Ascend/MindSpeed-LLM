from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature


class TrainingBasicFeature(MindSpeedFeature):
    
    def __init__(self):
        super(TrainingBasicFeature, self).__init__(feature_name="training", optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--jit-compile', action='store_true', default=False,
                           help='Setting jit compile mode to True')

    def register_patches(self, patch_manager, args):
        from mindspeed_llm.training.training import train
        from mindspeed_llm.training.checkpointing import load_checkpoint_wrapper
        from mindspeed_llm.tasks.posttrain.lora.utils import is_enable_qlora
        from mindspeed_llm.legacy.data import build_pretraining_data_loader
        from mindspeed_llm.training.utils import get_batch_on_this_tp_rank
        
        patch_manager.register_patch('megatron.training.training.build_pretraining_data_loader',
                          build_pretraining_data_loader)

        patch_manager.register_patch(
            'megatron.training.utils.get_batch_on_this_tp_rank', get_batch_on_this_tp_rank)
        

        from mindspeed_llm.training.training import get_model_wrapper
        patch_manager.register_patch('megatron.training.training.get_model', get_model_wrapper)
        
        patch_manager.register_patch('megatron.training.training.train', train)
        patch_manager.register_patch('megatron.training.training.load_checkpoint', load_checkpoint_wrapper)
   

