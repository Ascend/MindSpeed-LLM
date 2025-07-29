from typing import List

from mindspeed.deprecate import AutoExecuteFunction
from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.features_manager.tensor_parallel.unaligned_linear_feature import UnalignedLinearFeature
from mindspeed.features_manager.features_manager import MindSpeedFeaturesManager
from mindspeed.features_manager.fusions.fused_bias_swiglu import FusedSwigluFeature
from mindspeed.features_manager.fusions.fused_softmax import FusedSoftmaxFeature
from mindspeed.features_manager.fusions.fused_rope import FusedRoPEFeature
from mindspeed.features_manager.optimizer.virtual_optimizer import VirtualOptimizerFeature
from mindspeed.features_manager.optimizer.fused_ema_adamw_feature import FusedEmaAdamwFeature

from mindspeed_llm.features_manager.common.training import TrainingDefaultFeature
from mindspeed_llm.features_manager.common.rotary import RotaryPositionEmbeddingFeature
from mindspeed_llm.features_manager.common.embedding import LanguageModelEmbeddingFeature
from mindspeed_llm.features_manager.common.data import DataFeature
from mindspeed_llm.features_manager.common.moe_router import MOERouter
from mindspeed_llm.features_manager.models.mamba import MambaModel
from mindspeed_llm.features_manager.communication.coc import AscendCocFeature
from mindspeed_llm.features_manager.communication.gloo import DisableGlooFeature
from mindspeed_llm.features_manager.high_availability.high_availability import HighAvailabilityFeature
from mindspeed_llm.features_manager.transformer.mtp import MultiTokenPredictionFeature
from mindspeed_llm.features_manager.megatron_basic.megatron_basic import MegatronBasicFeature
from mindspeed_llm.features_manager.megatron_basic.requirements_basic import RequirementsBasicFeature
from mindspeed_llm.features_manager.megatron_basic.model_basic import ModelBasicFeature
from mindspeed_llm.features_manager.megatron_basic.training_basic import TrainingBasicFeature
from mindspeed_llm.features_manager.tokenizer.build_tokenizer import BuildTokenizerFeature
from mindspeed_llm.features_manager.dataset.dataset import DatasetFeature 
from mindspeed_llm.features_manager.finetune.finetune import FinetuneFeature
from mindspeed_llm.features_manager.finetune.lora import LoraFeature
from mindspeed_llm.features_manager.transformer.flash_attention.fusion_attention_feature import FusionAttentionFeature

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
    AscendCocFeature(),
    HighAvailabilityFeature(),
    MultiTokenPredictionFeature(),

    # MindSpeed-LLM Legacy Features
]


def add_megatron_basic_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        RequirementsBasicFeature(),
        MegatronBasicFeature(),
    ])


def add_llm_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        ModelBasicFeature(),
        TrainingBasicFeature(),
        RotaryPositionEmbeddingFeature(),
        DatasetFeature(),
        FinetuneFeature(),
        LoraFeature()
    ])


def add_fusions_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        FusedSwigluFeature(),
        FusedSoftmaxFeature(),
        FusedRoPEFeature(),
    ])


def add_optimizer_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        # Optimizer features: fused-ema-adamw
        FusedEmaAdamwFeature(),
        VirtualOptimizerFeature(),
    ])


def add_tokenizer_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        BuildTokenizerFeature()
    ])


def add_transformer_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        FusionAttentionFeature(),
    ])


def create_features_list():
    features_list = []
    add_megatron_basic_features(features_list)
    add_llm_features(features_list)
    add_fusions_features(features_list)
    add_optimizer_features(features_list)
    add_tokenizer_features(features_list)
    add_transformer_features(features_list)
    return features_list


@AutoExecuteFunction
def set_default_features_list():
    MindSpeedFeaturesManager.set_features_list(create_features_list())


set_default_features_list()