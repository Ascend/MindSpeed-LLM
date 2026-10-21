# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Compatibility scoped to the explicit AutoBridge training entrypoint.

Importing this module does not import Bridge, Torch, or install any patches.
The original LLM callables are restored when training returns or raises.
"""

from contextlib import contextmanager, ExitStack
from functools import wraps


@contextmanager
def _temporary_attribute(owner, name, value):
    original = vars(owner).get(name)
    existed = name in vars(owner)
    setattr(owner, name, value)
    try:
        yield
    finally:
        if existed:
            setattr(owner, name, original)
        else:
            delattr(owner, name)


def _state_dict_adapter(generate_state_dict, metadata_builder=None):
    @wraps(generate_state_dict)
    def generate(args, model, optimizer, scheduler, rng_state, *legacy_args, **kwargs):
        if metadata_builder is not None and args.ckpt_format == "torch_dist":
            # The old LLM save wrapper requests the removed flattened optimizer
            # format. Use the same metadata builder as native MCore save/load,
            # so optimizer sharding and model factories agree on the format.
            metadata = metadata_builder(args)
            for name in ("optim_sd_kwargs", "model_sd_kwargs"):
                state_kwargs = dict(kwargs.get(name) or {})
                state_kwargs.pop("sharding_type", None)
                state_kwargs["metadata"] = {**metadata, **(state_kwargs.get("metadata") or {})}
                kwargs[name] = state_kwargs
        # The unchanged LLM wrapper passes (iteration,) for torch and
        # (use_dist_ckpt, iteration) for torch_dist. Never forward that old
        # positional slot to MCore. The explicit keyword also works with
        # MindSpeed implementations retaining the deprecated slot.
        if len(legacy_args) == 2 and isinstance(legacy_args[0], bool):
            _, iteration = legacy_args
        elif len(legacy_args) == 1:
            iteration = legacy_args[0]
        elif not legacy_args:
            return generate_state_dict(args, model, optimizer, scheduler, rng_state, **kwargs)
        else:
            raise TypeError("Unexpected LLM checkpoint positional arguments")
        return generate_state_dict(
            args, model, optimizer, scheduler, rng_state, iteration=iteration, **kwargs
        )

    return generate


def validate_autobridge_training_args(args):
    """Reject combinations outside the synchronous DDP validation scope."""
    for name in ("use_torch_fsdp2", "use_megatron_fsdp", "use_custom_fsdp", "layerzero", "async_save"):
        if getattr(args, name, False):
            raise ValueError(
                f"AutoBridge online training has not validated --{name.replace('_', '-')}; "
                "use synchronous DDP training."
            )


@contextmanager
def autobridge_training_scope():
    """Adapt existing LLM callbacks only for one explicit AutoBridge run."""
    from megatron.core import mpu
    from megatron.training import get_args
    from megatron.training import checkpointing as mcore_checkpointing
    from mindspeed.features_manager.features_manager import MindSpeedFeaturesManager
    from mindspeed_llm.features_manager.transformer.mtp import MultiTokenPredictionFeature
    from mindspeed_llm.training import checkpointing, initialize

    mtp_features = [feature for feature in MindSpeedFeaturesManager.FEATURES_LIST
                    if isinstance(feature, MultiTokenPredictionFeature)]
    if len(mtp_features) != 1:
        raise RuntimeError("AutoBridge requires exactly one initialized LLM MTP feature")
    feature = mtp_features[0]
    pre_validate = feature.pre_validate_args
    warmup = initialize._warmup_jit_function  # pylint: disable=protected-access
    collect_rng = checkpointing.get_rng_state
    save_distributed = checkpointing.dist_checkpointing.save
    hidden_mtp_args = []

    def pre_validate_mtp(args):
        pre_validate(args)
        if args.mtp_num_layers and args.position_embedding_type == "mrope":
            # Keep the existing CP handling and post-validation restoration.
            hidden_mtp_args.append((args, args.mtp_num_layers))
            feature.origin_mtp_num_layers = args.mtp_num_layers
            args.mtp_num_layers = None

    def warmup_enabled_fusions():
        args = get_args()
        if args.bias_gelu_fusion or args.bias_swiglu_fusion or args.bias_dropout_fusion:
            warmup()

    def collect_rng_for_groups(ckpt_format, tp_group=None, pp_group=None):
        del tp_group, pp_group
        return collect_rng(
            ckpt_format,
            tp_group=mpu.get_tensor_model_parallel_group(),
            pp_group=mpu.get_pipeline_model_parallel_group(),
        )

    def save_with_content_metadata(*args, **kwargs):
        if "content_metadata" not in kwargs:
            metadata = mcore_checkpointing._build_sharded_state_dict_metadata(get_args())
            kwargs["content_metadata"] = mcore_checkpointing._clean_metadata_for_serialization(metadata)
        return save_distributed(*args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(_temporary_attribute(
            feature, "origin_mtp_num_layers", getattr(feature, "origin_mtp_num_layers", None)
        ))
        stack.enter_context(_temporary_attribute(feature, "pre_validate_args", pre_validate_mtp))
        stack.enter_context(_temporary_attribute(initialize, "_warmup_jit_function", warmup_enabled_fusions))
        stack.enter_context(_temporary_attribute(checkpointing, "get_rng_state", collect_rng_for_groups))
        stack.enter_context(_temporary_attribute(checkpointing.dist_checkpointing, "save", save_with_content_metadata))
        stack.enter_context(_temporary_attribute(
            checkpointing, "generate_state_dict", _state_dict_adapter(
                checkpointing.generate_state_dict, mcore_checkpointing._build_sharded_state_dict_metadata
            )
        ))
        try:
            yield
        finally:
            # If core validation raises before post_validate_args, do not leave
            # the caller's Namespace with its requested MTP depth hidden.
            for args, mtp_num_layers in hidden_mtp_args:
                args.mtp_num_layers = mtp_num_layers
