# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from argparse import ArgumentParser, Namespace
from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager


class NumLayerListFeature(MindSpeedFeature):

    def __init__(self):
        super(NumLayerListFeature, self).__init__(feature_name="num-layer-list", optimization_level=2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--num-layer-list',
                           type=str, help='a list of number of layers, '
                                          'seperated by comma; e.g., 4,4,4,4')

    def pre_validate_args(self, args: Namespace):
        self.origin_num_layers = None
        if args.num_layer_list:
            self.origin_num_layers = args.num_layers
            args.num_layers = len(args.num_layer_list.split(','))

    def validate_args(self, args: Namespace):
        if args.num_layer_list:
            if getattr(args, 'save_model_type', None) != 'hf':
                if len(args.num_layer_list.split(',')) != args.pipeline_model_parallel_size:
                    raise ValueError("len(args.num_layer_list) != args.pipeline_model_parallel_size")
            if not args.pipeline_model_parallel_size > 1:
                raise ValueError("Dynamic pipeline model should work with pipeline parallel.")
            if args.num_layers_per_virtual_pipeline_stage:
                raise ValueError("Dynamic pipeline model and virtual pipeline cannot be enabled at the same time.")
            if args.schedules_method == "dualpipev":
                raise ValueError("Dynamic pipeline model and dualpipev cannot be enabled at the same time.")
            if args.noop_layers:
                from mindspeed_llm.training.utils import print_rank0_by_args
                print_rank0_by_args("num layer list would be disabled when noop-layer is activated.")
                args.num_layer_list = None

    def post_validate_args(self, args: Namespace):
        if self.origin_num_layers:
            args.num_layers = self.origin_num_layers
            args.encoder_num_layers = self.origin_num_layers

    def register_patches(
            self,
            patch_manager: MindSpeedPatchesManager,
            args: Namespace,
    ):
        if args.num_layer_list:
            from mindspeed_llm.core import get_num_layers_to_build
            from mindspeed_llm.core.transformer.transformer_block import get_layer_offset_wrapper
            from mindspeed_llm.training.arguments import core_transformer_config_from_args_wrapper
            patch_manager.register_patch('megatron.core.transformer.transformer_block.get_num_layers_to_build',
                                         get_num_layers_to_build)
            patch_manager.register_patch('megatron.core.transformer.transformer_layer.TransformerLayer._get_layer_offset',
                                         get_layer_offset_wrapper)
            patch_manager.register_patch('megatron.training.arguments.core_transformer_config_from_args',
                                         core_transformer_config_from_args_wrapper)