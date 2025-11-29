from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature


class ModelBasicFeature(MindSpeedFeature):

    def __init__(self):
        super(ModelBasicFeature, self).__init__(feature_name="model", optimization_level=0)

    def register_patches(self, patch_manager, args):
        self.patch_model_patches(patch_manager, args)

    def patch_model_patches(self, pm, args):
        from mindspeed_llm.training.tokenizer import build_tokenizer
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
        from mindspeed_llm.core.models.gpt.gpt_model import GPTModel
        from mindspeed_llm.training.utils import get_device_wrapper
        from mindspeed_llm.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec_wrapper
        from mindspeed_llm.core import vocab_parallel_embedding_forward, vocab_embedding_init_func, checkpoint_forward_wrapper, \
            checkpoint_backward_wrapper
        pm.register_patch('megatron.training.global_vars.build_tokenizer',
                           build_tokenizer)
        pm.register_patch('megatron.core.models.gpt.gpt_model.GPTModel',
                           GPTModel)

        pm.register_patch('megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward',
                           vocab_parallel_embedding_forward)
        pm.register_patch('megatron.core.tensor_parallel.layers.VocabParallelEmbedding.__init__',
                           vocab_embedding_init_func)
        pm.register_patch('megatron.core.tensor_parallel.random.CheckpointFunction.forward',
                           checkpoint_forward_wrapper)
        pm.register_patch('megatron.core.tensor_parallel.random.CheckpointFunction.backward',
                           checkpoint_backward_wrapper)

        # Layer Definition
        pm.register_patch('megatron.core.models.gpt.gpt_layer_specs.get_gpt_layer_local_spec',
                           get_gpt_layer_local_spec_wrapper)
        pm.register_patch('megatron.training.dist_signal_handler.get_device',
                           get_device_wrapper)