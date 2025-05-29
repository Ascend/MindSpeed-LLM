from mindspeed.features_manager.tensor_parallel.unaligned_linear_feature import UnalignedLinearFeature

from mindspeed_llm.features_manager.common.training import TrainingDefaultFeature
from mindspeed_llm.features_manager.communication.gloo import DisableGlooFeature
from mindspeed_llm.features_manager.common.rotary import RotaryPositionEmbeddingFeature
from mindspeed_llm.features_manager.common.embedding import LanguageModelEmbeddingFeature
from mindspeed_llm.features_manager.models.mamba import MambaModel

from mindspeed_llm.features_manager.communication.coc import AscendCocFeature
from mindspeed_llm.features_manager.common.data import DataFeature
from mindspeed_llm.features_manager.common.moe_router import MOERouter

FEATURES_LIST = [
    # MindSpeed Legacy Features
    
    # MindSpeed Mcore Features
    UnalignedLinearFeature(),

    
    # MindSpeed-LLM Mcore Features
    TrainingDefaultFeature(),
    DataFeature(),
    DisableGlooFeature(),
    RotaryPositionEmbeddingFeature(),
    LanguageModelEmbeddingFeature(),
    MambaModel(),
    MOERouter(),
    AscendCocFeature()

    # MindSpeed-LLM Legacy Features
]