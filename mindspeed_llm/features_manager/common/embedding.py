from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature


class LanguageModelEmbeddingFeature(MindSpeedFeature):
    def __init__(self):
        super(LanguageModelEmbeddingFeature, self).__init__(feature_name="language-model-embedding", optimization_level=0)

    def register_patches(self, patch_manager, args):
        from mindspeed.core.models.common.embeddings.language_model_embedding import language_model_embedding_forward_wrapper
        patch_manager.register_patch('megatron.core.models.common.embeddings.language_model_embedding.LanguageModelEmbedding.forward',
                                      language_model_embedding_forward_wrapper)