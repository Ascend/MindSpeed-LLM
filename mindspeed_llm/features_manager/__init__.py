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
    ContextParallelKvCacheFeature,
    TorchFullyShardedDataParallelFeature,
    ProfilerDefaultFeature,
    OptimizeP2PCommFeature,
    FusionAttentionV2Feature
)
from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.features_manager.features_manager import MindSpeedFeaturesManager

from mindspeed_llm.features_manager.affinity.affinity import AffinityFeature
from mindspeed_llm.features_manager.context_parallel.context_parallel_feature import ContextParallelFeature
from mindspeed_llm.features_manager.context_parallel.ulysses_context_parallel import UlyssesContextParallelFeature
from mindspeed_llm.features_manager.context_parallel.mamba_context_parallel import MambaContextParallelFeature
from mindspeed_llm.features_manager.common.data import DataFeature
from mindspeed_llm.features_manager.models.module import ModuleFeature
from mindspeed_llm.features_manager.common.embedding import LanguageModelEmbeddingFeature
from mindspeed_llm.features_manager.common.rotary import RotaryPositionEmbeddingFeature
from mindspeed_llm.features_manager.common.training import TrainingDefaultFeature
from mindspeed_llm.features_manager.tensor_parallel.coc import CoCFeature
from mindspeed_llm.features_manager.dataset.dataset import DatasetFeature
from mindspeed_llm.features_manager.finetune.finetune import FinetuneFeature
from mindspeed_llm.features_manager.dataset.data_preprocess import DatasetPreprocessFeature
from mindspeed_llm.features_manager.finetune.lora import LoraFeature
from mindspeed_llm.features_manager.finetune.lu_lora import LuLoraFeature
from mindspeed_llm.features_manager.high_availability.high_availability import HighAvailabilityFeature
from mindspeed_llm.features_manager.inference.inference import InferenceFeature
from mindspeed_llm.features_manager.evaluation.evaluation import EvaluationFeature
from mindspeed_llm.features_manager.megatron_basic.megatron_basic import MegatronBasicFeature
from mindspeed_llm.features_manager.megatron_basic.model_basic import ModelBasicFeature
from mindspeed_llm.features_manager.megatron_basic.requirements_basic import RequirementsBasicFeature
from mindspeed_llm.features_manager.megatron_basic.training_basic import TrainingBasicFeature
from mindspeed_llm.features_manager.models.mamba import MambaModel
from mindspeed_llm.features_manager.transformer.multi_latent_attention.dsa_indexer_feature import DSAIndexerFeature
from mindspeed_llm.features_manager.moe.moe_router import MoERouter
from mindspeed_llm.features_manager.moe.shared_expert import MoESharedExpertsFeature
from mindspeed_llm.features_manager.moe.tp_extend_ep import MoETpExtendEpFeature
from mindspeed_llm.features_manager.dpo.dpo import DPOFeature
from mindspeed_llm.features_manager.pipeline_parallel.dualpipev_feature import DualpipeVFeature
from mindspeed_llm.features_manager.pipeline_parallel.noop_layers import NoopLayersFeature
from mindspeed_llm.features_manager.functional.profiling import ProfilingFeature
from mindspeed_llm.features_manager.tokenizer.build_tokenizer import BuildTokenizerFeature
from mindspeed_llm.features_manager.transformer.flash_attention.fusion_attention_feature import FusionAttentionFeature
from mindspeed_llm.features_manager.transformer.flash_attention.alibi_feature import AlibiFeature
from mindspeed_llm.features_manager.transformer.flash_attention.reset_attention_mask_feature import ResetAttentionMaskFeature
from mindspeed_llm.features_manager.transformer.mtp import MultiTokenPredictionFeature
from mindspeed_llm.features_manager.transformer.multi_latent_attention.mla_feature import MLAFeature
from mindspeed_llm.features_manager.transformer.qwen3_next_attention.qwen3_next_feature import Qwen3NextFeature
from mindspeed_llm.features_manager.transformer.transformer_block import TransformerBlockFeature
from mindspeed_llm.features_manager.pipeline_parallel.num_layer_list import NumLayerListFeature
from mindspeed_llm.features_manager.ai_framework.ms_patch_feature import MindSporePatchFeature
from mindspeed_llm.features_manager.tensor_parallel.tp_2d import TP2dFeature
from mindspeed_llm.features_manager.arguments.deprecated_args import DeprecatedArgsFeature
from mindspeed_llm.features_manager.convert_checkpoint.convert_checkpoint import CheckpointFeature
from mindspeed_llm.features_manager.fsdp2.fsdp2 import FSDP2Feature


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
    CoCFeature(),
    HighAvailabilityFeature(),
    MultiTokenPredictionFeature(),

    # MindSpeed-LLM Legacy Features
]


def add_megatron_basic_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        RequirementsBasicFeature(),
        MegatronBasicFeature(),
        TransformerEngineBasicFeature(),
        Qwen3NextFeature(),        
    ])


def add_llm_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        ModelBasicFeature(),
        TrainingBasicFeature(),
        DatasetFeature(),
        DatasetPreprocessFeature(),
        ModuleFeature(),
        NumLayerListFeature(),
        DPOFeature(),
        InferenceFeature(),
        EvaluationFeature(),
        DeprecatedArgsFeature(),
        DSAIndexerFeature(),
        MambaModel(),
        LanguageModelEmbeddingFeature(),
        CheckpointFeature(),
        FSDP2Feature()
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
        CoCFeature(),
        MC2Feature(),
        TP2dFeature()
    ])


def add_pipeline_parallel_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        RiPipeSchedulesBubbleFeature(),
        RiPipeSchedulesAdvanceFeature(),
        NoopLayersFeature(),
        OptimizeP2PCommFeature(),
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
        # LLM feature
        AlibiFeature(),
        # LLM feature
        ResetAttentionMaskFeature(),
        FusionAttentionV2Feature()
    ])


def add_tokenizer_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        BuildTokenizerFeature()
    ])


def add_distributed_features(features_list: List[MindSpeedFeature]): 
    features_list.extend([ 
        TorchFullyShardedDataParallelFeature() 
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
        ProfilingFeature(),
        NPUDeterministicFeature(),
        ProfilerDefaultFeature()
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


def add_high_availability_feature(features_list: List[MindSpeedFeature]):
    features_list.extend([
        HighAvailabilityFeature()
    ])


def add_finetune_feature(features_list: List[MindSpeedFeature]):
    features_list.extend([
        FinetuneFeature(),
        LoraFeature(),
        LuLoraFeature(),
    ])


def add_ai_framework_feature(features_list: List[MindSpeedFeature]):
    features_list.extend([
        MindSporePatchFeature(),
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
    add_distributed_features(features_list)
    add_reuse_param_features(features_list)
    add_swap_manage_features(features_list)
    add_moe_features(features_list)
    add_hccl_buffer_features(features_list)
    add_optimizer_features(features_list)
    add_swap_optimizer_feature(features_list)
    add_disable_gloo_group_feature(features_list)
    add_high_availability_feature(features_list)
    add_finetune_feature(features_list)
    add_ai_framework_feature(features_list)
    return features_list


@AutoExecuteFunction
def set_default_features_list():
    MindSpeedFeaturesManager.set_features_list(create_features_list())


set_default_features_list()
