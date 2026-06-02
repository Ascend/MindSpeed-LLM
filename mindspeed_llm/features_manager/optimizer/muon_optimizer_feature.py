# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

"""Feature gate for backporting Muon optimizer support to Megatron 0.12.x."""

import warnings
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


def _add_choice(parser: ArgumentParser, option: str, value: str) -> None:
    for action in parser._actions:
        if option in action.option_strings and action.choices is not None:
            if value not in action.choices:
                action.choices = list(action.choices) + [value]
            return


def _add_apply_wd_to_qk_layernorm_arg(group, parser: ArgumentParser) -> None:
    for action in parser._actions:
        if "--apply-wd-to-qk-layernorm" in action.option_strings:
            return
    group.add_argument(
        "--apply-wd-to-qk-layernorm",
        action="store_true",
        help="Apply weight decay to qk layernorm as a special case.",
    )


class MuonOptimizerFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__("muon-optimizer", optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        _add_choice(parser, "--optimizer", "muon")
        _add_choice(parser, "--optimizer", "dist_muon")

        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            "--muon-momentum",
            type=float,
            default=0.95,
            help="Momentum factor for Muon optimizer.",
        )
        group.add_argument(
            "--muon-no-split-qkv",
            action="store_false",
            default=True,
            dest="muon_split_qkv",
            help="Whether to split QKV parameters for Muon optimizer.",
        )
        group.add_argument(
            "--muon-nesterov",
            action="store_true",
            help="Whether to use Nesterov-style momentum in Muon.",
        )
        group.add_argument(
            "--muon-scale-mode",
            type=str,
            default="spectral",
            choices=["spectral", "unit_rms_norm", "shape_scaling"],
            help="Scale mode for Muon optimizer.",
        )
        group.add_argument(
            "--muon-fp32-matmul-prec",
            type=str,
            default="medium",
            choices=["low", "medium", "high"],
            help="FP32 matmul precision for Newton-Schulz iteration.",
        )
        group.add_argument(
            "--muon-coefficient-type",
            type=str,
            default="quintic",
            help="Newton-Schulz coefficient type for Muon optimizer.",
        )
        group.add_argument(
            "--muon-num-ns-steps",
            type=int,
            default=5,
            help="Number of Newton-Schulz steps for Muon optimizer.",
        )
        group.add_argument(
            "--muon-tp-mode",
            type=str,
            default="blockwise",
            choices=["blockwise", "duplicated", "distributed"],
            help="How to perform NS calculation for tensor-parallel weights.",
        )
        group.add_argument(
            "--muon-extra-scale-factor",
            type=float,
            default=1.0,
            help="Additional scale factor for the Muon update.",
        )
        group.add_argument(
            "--muon-scalar-optimizer",
            type=str,
            default="adam",
            choices=["adam", "lion"],
            help="Optimizer for scalar/non-matrix params when using Muon.",
        )
        _add_apply_wd_to_qk_layernorm_arg(group, parser)

    def post_validate_args(self, args):
        if getattr(args, "optimizer", None) == "dist_muon":
            warnings.warn(
                "optimizer='dist_muon' is deprecated. Use --optimizer muon --use-distributed-optimizer instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            args.optimizer = "muon"
            args.use_layer_wise_distributed_optimizer = True

        if getattr(args, "optimizer", None) == "muon" and getattr(args, "use_distributed_optimizer", False):
            args.use_layer_wise_distributed_optimizer = True
            args.use_distributed_optimizer = False

        if getattr(args, "optimizer", None) == "muon" and getattr(
            args, "overlap_param_gather_with_optimizer_step", False
        ):
            warnings.warn(
                "MindSpeed Muon on Megatron 0.12.1 does not support "
                "--overlap-param-gather-with-optimizer-step; disabling it for this path.",
                UserWarning,
                stacklevel=2,
            )
            args.overlap_param_gather_with_optimizer_step = False

    def validate_args(self, args):
        if getattr(args, "optimizer", None) != "muon":
            return
        if getattr(args, "fp16", False):
            raise AssertionError("Muon optimizer does not support fp16; use bf16 or fp32.")
        if getattr(args, "use_custom_fsdp", False) or getattr(args, "use_torch_fsdp2", False):
            raise AssertionError("Muon optimizer does not support FSDP in this MindSpeed patch.")
        if getattr(args, "overlap_param_gather", False):
            if not getattr(args, "use_layer_wise_distributed_optimizer", False):
                raise AssertionError(
                    "Muon optimizer requires layer-wise distributed optimizer when "
                    "--overlap-param-gather is enabled. Use --use-distributed-optimizer "
                    "or optimizer='dist_muon'."
                )
            if not getattr(args, "overlap_grad_reduce", False):
                raise AssertionError("Must use --overlap-param-gather with --overlap-grad-reduce.")

    def register_patches(self, patch_manager, args):
        if getattr(args, "optimizer", None) not in ("muon", "dist_muon"):
            return

        from mindspeed.core.optimizer.muon.adaptor import (
            add_muon_tensor_model_parallel_attributes,
            chained_optimizer_count_zeros,
            copy_muon_tensor_model_parallel_attributes_wrapper,
            count_zeros_fp32,
            get_megatron_optimizer_muon_wrapper,
            get_megatron_optimizer_based_on_param_groups_wrapper,
            get_main_grads_for_grad_norm,
            megatron_optimizer_count_zeros,
            param_is_not_tensor_parallel_duplicate,
        )
        from mindspeed.core.optimizer.muon.muon import get_megatron_muon_optimizer
        from mindspeed.core.optimizer.muon.optimizer_config import optimizer_config_init_wrapper
        from mindspeed.core.optimizer.muon.checkpointing import (
            load_base_checkpoint_layer_wise_optimizer_wrapper,
            load_checkpoint_layer_wise_optimizer_wrapper,
            save_checkpoint_layer_wise_optimizer_wrapper,
        )

        add_muon_tensor_model_parallel_attributes()
        patch_manager.register_patch(
            "megatron.core.optimizer.get_megatron_optimizer",
            get_megatron_optimizer_muon_wrapper,
        )
        patch_manager.register_patch(
            "megatron.core.optimizer._get_megatron_optimizer_based_on_param_groups",
            get_megatron_optimizer_based_on_param_groups_wrapper,
        )
        patch_manager.register_patch(
            "megatron.core.optimizer.optimizer_config.OptimizerConfig.__init__",
            optimizer_config_init_wrapper,
        )
        patch_manager.register_patch(
            "megatron.core.tensor_parallel.layers.copy_tensor_model_parallel_attributes",
            copy_muon_tensor_model_parallel_attributes_wrapper,
        )
        patch_manager.register_patch(
            "megatron.core.optimizer.muon.get_megatron_muon_optimizer",
            get_megatron_muon_optimizer,
            create_dummy=True,
        )
        patch_manager.register_patch(
            "megatron.core.tensor_parallel.layers.param_is_not_tensor_parallel_duplicate",
            param_is_not_tensor_parallel_duplicate,
        )
        patch_manager.register_patch(
            "megatron.core.optimizer.optimizer.MegatronOptimizer.get_main_grads_for_grad_norm",
            get_main_grads_for_grad_norm,
        )
        patch_manager.register_patch(
            "megatron.core.optimizer.clip_grads.count_zeros_fp32",
            count_zeros_fp32,
        )
        patch_manager.register_patch(
            "megatron.core.optimizer.optimizer.MegatronOptimizer.count_zeros",
            megatron_optimizer_count_zeros,
        )
        patch_manager.register_patch(
            "megatron.core.optimizer.optimizer.ChainedOptimizer.count_zeros",
            chained_optimizer_count_zeros,
        )
        patch_manager.register_patch(
            "megatron.training.checkpointing.save_checkpoint",
            save_checkpoint_layer_wise_optimizer_wrapper,
        )
        patch_manager.register_patch(
            "megatron.training.checkpointing._load_base_checkpoint",
            load_base_checkpoint_layer_wise_optimizer_wrapper,
        )
        patch_manager.register_patch(
            "megatron.training.checkpointing.load_checkpoint",
            load_checkpoint_layer_wise_optimizer_wrapper,
        )
        from mindspeed.core.optimizer.muon.param_and_grad_buffer import (
            distributed_data_parallel_start_param_sync_wrapper,
            finish_grad_sync_wrapper,
            param_and_grad_bucket_group_init_wrapper,
            set_layerwise_params_list,
        )

        patch_manager.register_patch(
            "megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucket.set_layerwise_params_list",
            set_layerwise_params_list,
            create_dummy=True,
        )
        patch_manager.register_patch(
            "megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.__init__",
            param_and_grad_bucket_group_init_wrapper,
        )
        patch_manager.register_patch(
            "megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.finish_grad_sync",
            finish_grad_sync_wrapper,
        )
        patch_manager.register_patch(
            "megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.start_param_sync",
            distributed_data_parallel_start_param_sync_wrapper,
        )
