from typing import List

from mindspeed.deprecate import AutoExecuteFunction
from mindspeed.features_manager import (
    DisableGlooGroupFeature,
    FusedEmaAdamwFeature,
    FusedMoEPermuteFeature,
    FusedSoftmaxFeature,
    FusedSwigluFeature,
    GroupedMatmulFeature,
    MC2Feature,
    MoEAlltoAllOverLapFeature,
    MoEAllGatherOverLapFeature,
    MoEFwdBwdOverlapFeature,
    MoEGmmFeature,
    MoEZeroMemoryFeature,
    OptimizeSendRecvCommFeature,
    SwapOptimizerFeature,
    ReuseFP32Param,
    RiPipeSchedulesAdvanceFeature,
    RiPipeSchedulesBubbleFeature,
    TransformerEngineBasicFeature,
    UnalignedLinearFeature,
    UnalignedPipelineFeature,
    VirtualOptimizerFeature,
    HcclBufferAdaptiveFeature,
    HcclBufferSetFeature,
    RecomputeNormFeature,
    RecomputeActivationFeature,
    NPUDeterministicFeature,
    EnableRecomputeLayersPerPPRank,
    RecomputeMethodFeature,
    SmartSwapFeature,
    SwapAttentionFeature,
    ContextParallelKvCacheFeature
)
from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.features_manager.features_manager import MindSpeedFeaturesManager

from mindspeed_llm.features_manager.affinity.affinity import AffinityFeature
from mindspeed_llm.features_manager.context_parallel.context_parallel_feature import ContextParallelFeature
from mindspeed_llm.features_manager.context_parallel.ulysses_context_parallel import UlyssesContextParallelFeature
from mindspeed_llm.features_manager.context_parallel.mamba_context_parallel import MambaContextParallelFeature
from mindspeed_llm.features_manager.common.data import DataFeature
from mindspeed_llm.features_manager.common.embedding import LanguageModelEmbeddingFeature
from mindspeed_llm.features_manager.common.rotary import RotaryPositionEmbeddingFeature
from mindspeed_llm.features_manager.common.training import TrainingDefaultFeature
from mindspeed_llm.features_manager.communication.coc import AscendCocFeature
from mindspeed_llm.features_manager.dataset.dataset import DatasetFeature
from mindspeed_llm.features_manager.finetune.finetune import FinetuneFeature
from mindspeed_llm.features_manager.finetune.lora import LoraFeature
from mindspeed_llm.features_manager.finetune.lu_lora import LuLoraFeature
from mindspeed_llm.features_manager.high_availability.high_availability import HighAvailabilityFeature
from mindspeed_llm.features_manager.megatron_basic.megatron_basic import MegatronBasicFeature
from mindspeed_llm.features_manager.megatron_basic.model_basic import ModelBasicFeature
from mindspeed_llm.features_manager.megatron_basic.requirements_basic import RequirementsBasicFeature
from mindspeed_llm.features_manager.megatron_basic.training_basic import TrainingBasicFeature
from mindspeed_llm.features_manager.models.mamba import MambaModel
from mindspeed_llm.features_manager.moe.moe_router import MoERouter
from mindspeed_llm.features_manager.moe.shared_expert import MoESharedExpertsFeature
from mindspeed_llm.features_manager.moe.tp_extend_ep import MoETpExtendEpFeature
from mindspeed_llm.features_manager.pipeline_parallel.dualpipev_feature import DualpipeVFeature
from mindspeed_llm.features_manager.pipeline_parallel.noop_layers import NoopLayersFeature
from mindspeed_llm.features_manager.tokenizer.build_tokenizer import BuildTokenizerFeature
from mindspeed_llm.features_manager.transformer.flash_attention.fusion_attention_feature import FusionAttentionFeature
from mindspeed_llm.features_manager.transformer.mtp import MultiTokenPredictionFeature
from mindspeed_llm.features_manager.transformer.multi_latent_attention.mla_feature import MLAFeature
from mindspeed_llm.features_manager.transformer.transformer_block import TransformerBlockFeature


FEATURES_LIST = [
    # MindSpeed Legacy Features

    # MindSpeed Mcore Features
    UnalignedLinearFeature(),
    # MindSpeed-LLM Mcore Features
    TrainingDefaultFeature(),
    DataFeature(),
    LoraFeature(),
    DisableGlooGroupFeature(),
    RotaryPositionEmbeddingFeature(),
    LanguageModelEmbeddingFeature(),
    MambaModel(),
    MoERouter(),
    AscendCocFeature(),
    HighAvailabilityFeature(),
    MultiTokenPredictionFeature(),

    # MindSpeed-LLM Legacy Features
]


def add_megatron_basic_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        RequirementsBasicFeature(),
        MegatronBasicFeature(),
        TransformerEngineBasicFeature(),
    ])


def add_llm_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        ModelBasicFeature(),
        TrainingBasicFeature(),
        DatasetFeature(),
        FinetuneFeature(),
        LoraFeature(),
        LuLoraFeature(),
        MambaModel(),
        LanguageModelEmbeddingFeature(),
    ])


def add_affinity_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        AffinityFeature(),
    ])


def add_context_parallel_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        ContextParallelFeature(),
        UlyssesContextParallelFeature(),
        ContextParallelKvCacheFeature(),
        MambaContextParallelFeature()
    ])


def add_fusions_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        FusedSwigluFeature(),
        FusedSoftmaxFeature(),
        RotaryPositionEmbeddingFeature(),
        GroupedMatmulFeature(),
        FusedMoEPermuteFeature(),
    ])


def add_tensor_parallel_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        AscendCocFeature(),
        MC2Feature()
    ])


def add_pipeline_parallel_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        RiPipeSchedulesBubbleFeature(),
        RiPipeSchedulesAdvanceFeature(),
        NoopLayersFeature(),
        OptimizeSendRecvCommFeature(),
        UnalignedPipelineFeature(),
        DualpipeVFeature(),
    ])


def add_transformer_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        FusionAttentionFeature(),
        # LLM feature
        MLAFeature(),
        # LLM feature
        MultiTokenPredictionFeature(),
        # LLM feature
        TransformerBlockFeature(),
    ])


def add_tokenizer_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        BuildTokenizerFeature()
    ])


def add_reuse_param_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        ReuseFP32Param()
    ])


def add_swap_manage_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        SmartSwapFeature(),
        SwapAttentionFeature()
    ])


def add_moe_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        MoEGmmFeature(),
        # LLM feature
        MoERouter(),
        MoETpExtendEpFeature(),
        # LLM feature
        MoESharedExpertsFeature(),
        MoEAllGatherOverLapFeature(),
        MoEFwdBwdOverlapFeature(),
        MoEAlltoAllOverLapFeature(),
        MoEZeroMemoryFeature(),
    ])


def add_hccl_buffer_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        HcclBufferSetFeature(),
        HcclBufferAdaptiveFeature(),
    ])


def add_optimizer_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        FusedEmaAdamwFeature(),
        VirtualOptimizerFeature(),
    ])


def add_functional_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        NPUDeterministicFeature(),
    ])


def add_recompute_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        RecomputeActivationFeature(),
        RecomputeNormFeature(),
        EnableRecomputeLayersPerPPRank(),
        RecomputeMethodFeature()
    ])


def add_swap_optimizer_feature(features_list: List[MindSpeedFeature]):
    features_list.extend([
        SwapOptimizerFeature(),
    ])


def add_disable_gloo_group_feature(features_list: List[MindSpeedFeature]):
    features_list.extend([
        DisableGlooGroupFeature()
    ])


def create_features_list():
    features_list = []
    add_megatron_basic_features(features_list)
    add_context_parallel_features(features_list)
    add_llm_features(features_list)
    add_affinity_features(features_list)
    add_fusions_features(features_list)
    add_recompute_features(features_list)
    add_functional_features(features_list)
    add_tensor_parallel_features(features_list)
    add_pipeline_parallel_features(features_list)
    add_transformer_features(features_list)
    add_tokenizer_features(features_list)
    add_reuse_param_features(features_list)
    add_swap_manage_features(features_list)
    add_moe_features(features_list)
    add_hccl_buffer_features(features_list)
    add_optimizer_features(features_list)
    add_swap_optimizer_feature(features_list)
    add_disable_gloo_group_feature(features_list)
    return features_list


@AutoExecuteFunction
def set_default_features_list():
    MindSpeedFeaturesManager.set_features_list(create_features_list())


set_default_features_list()
