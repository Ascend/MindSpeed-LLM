from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature


class ModelBasicFeature(MindSpeedFeature):

    def __init__(self):
        super(ModelBasicFeature, self).__init__(feature_name="model", optimization_level=0)
    
    def register_patches(self, patch_manager, args):
        self.patch_attention_patches(patch_manager, args)
        self.patch_model_patches(patch_manager, args)

    def patch_model_patches(self, pm, args):
        from mindspeed_llm.training.tokenizer import build_tokenizer
        from mindspeed_llm.core.models.gpt.gpt_model import GPTModel
        from mindspeed_llm.core.models.common.embeddings.language_model_embedding import \
            language_model_embedding_init_func
        from mindspeed_llm.core import vocab_parallel_embedding_forward, vocab_embedding_init_func, checkpoint_forward_wrapper, \
            checkpoint_backward_wrapper
        pm.register_patch('megatron.training.global_vars.build_tokenizer', build_tokenizer)
        pm.register_patch('megatron.core.models.gpt.gpt_model.GPTModel', GPTModel)
        pm.register_patch(
            'megatron.core.models.common.embeddings.language_model_embedding.LanguageModelEmbedding.__init__',
            language_model_embedding_init_func)

        pm.register_patch('megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward',
                                    vocab_parallel_embedding_forward)
        pm.register_patch('megatron.core.tensor_parallel.layers.VocabParallelEmbedding.__init__',
                                    vocab_embedding_init_func)
        pm.register_patch('megatron.core.tensor_parallel.random.CheckpointFunction.forward',
                                    checkpoint_forward_wrapper)
        pm.register_patch('megatron.core.tensor_parallel.random.CheckpointFunction.backward',
                                    checkpoint_backward_wrapper)

    def patch_attention_patches(self, pm, args):
        from mindspeed_llm.core.transformer.dot_product_attention import dot_product_attention_init, \
            dot_product_attention_forward_wrapper, ulysses_context_parallel_forward_wrapper
        pm.register_patch('megatron.core.transformer.dot_product_attention.DotProductAttention.__init__',
                          dot_product_attention_init)
        pm.register_patch('megatron.core.transformer.dot_product_attention.DotProductAttention.forward',
                          dot_product_attention_forward_wrapper)
        pm.register_patch(
            'megatron.core.transformer.custom_layers.transformer_engine.TEDotProductAttention.__init__',
            dot_product_attention_init)
        pm.register_patch(
            'megatron.core.transformer.custom_layers.transformer_engine.TEDotProductAttention.forward',
            dot_product_attention_forward_wrapper)


    
