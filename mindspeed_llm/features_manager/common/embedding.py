# Copyright 2024 Huawei Technologies Co., Ltd.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature


class LanguageModelEmbeddingFeature(MindSpeedFeature):
    def __init__(self):
        super(LanguageModelEmbeddingFeature, self).__init__(feature_name="language-model-embedding", optimization_level=0)

    def register_patches(self, patch_manager, args):
        from mindspeed.core.models.common.embeddings.language_model_embedding import language_model_embedding_forward_wrapper
        from mindspeed_llm.core.models.common.embeddings.language_model_embedding import language_model_embedding_init_func

        patch_manager.register_patch('megatron.core.models.common.embeddings.language_model_embedding.LanguageModelEmbedding.__init__',
                                      language_model_embedding_init_func)
