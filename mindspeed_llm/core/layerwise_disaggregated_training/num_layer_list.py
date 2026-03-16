# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

from typing import Union

from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    TransformerBlock,
)
from megatron.core import parallel_state
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    BaseTransformerLayer,
)
from megatron.training import get_args


try:
    from megatron.core.extensions.transformer_engine import (
        TENorm,
        get_cpu_offload_context,
        te_checkpoint,
    )

    HAVE_TE = True
    LayerNormImpl = TENorm
except ImportError:
    HAVE_TE = False
    get_cpu_offload_context = None

    try:
        import apex  # pylint: disable=unused-import
        LayerNormImpl = FusedLayerNorm
    except ImportError:
        from megatron.core.transformer.torch_norm import WrappedTorchNorm
        LayerNormImpl = WrappedTorchNorm


def _get_block_submodules(
    config: TransformerConfig, spec: Union[TransformerBlockSubmodules, ModuleSpec]
) -> TransformerBlockSubmodules:
    """
    Retrieve or construct TransformerBlockSubmodules based on the provided specification.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        spec (Union[TransformerBlockSubmodules, ModuleSpec]): Specification for the
            transformer block submodules. Can be either a TransformerBlockSubmodules
            instance or a ModuleSpec.

    Returns:
        TransformerBlockSubmodules: The submodules for the transformer block.
    """

    # Transformer block submodules.
    if isinstance(spec, TransformerBlockSubmodules):
        return spec

    # ModuleSpec here is generally assumed to be for a transformer layer that
    # is implemented in `transformer_layer.py` or if it subclasses
    # `BaseTransformerLayer` from the `transformer_layer.py` file.
    elif isinstance(spec, ModuleSpec):
        if issubclass(spec.module, TransformerBlock):
            return spec.submodules
        elif issubclass(spec.module, BaseTransformerLayer):
            num_layers = get_num_layers_to_build(config)
            return TransformerBlockSubmodules(
                layer_specs=[spec] * num_layers, layer_norm=LayerNormImpl
            )
        else:
            raise Exception(f"specialize for {spec.module.__name__}.")
    else:
        raise Exception(f"specialize for {type(spec).__name__}.")


def get_num_layers_to_build(config: TransformerConfig) -> int:
    num_layers_per_pipeline_rank = (
        config.num_layers // parallel_state.get_pipeline_model_parallel_world_size()
    )

    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
        # Interleaved pipeline parallelism:
        # Number of layers in each model chunk is the number of layers in the stage,
        # divided by the number of model chunks in a stage.
        # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0]  [2]  [4]  [6]
        # Stage 1: [1]  [3]  [5]  [7]
        # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0, 1]  [4, 5]
        # Stage 1: [2, 3]  [6, 7]

        vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

        num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size

        num_layers_to_build = num_layers_per_virtual_rank

    else:
        # Non-interleaved pipeline parallelism:
        # Each stage gets a contiguous set of layers.

        num_layers_to_build = num_layers_per_pipeline_rank

    num_layer_list = config.num_layer_list
    if num_layer_list:
        pp_stage = parallel_state.get_pipeline_model_parallel_rank()
        num_layers_to_build = num_layer_list[pp_stage]

        # add: layerwise_disaggregated_training
        # U-shaped split scenario, the number of layers to build is determined by
        # both pp stage and vpp rank
        args = get_args()
        if getattr(args, "layerwise_disaggregated_training", None):
            vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
            num_layers_to_build = num_layer_list[pp_stage][vpp_rank]

    return num_layers_to_build