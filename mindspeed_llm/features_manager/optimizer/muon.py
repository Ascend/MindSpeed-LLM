# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
from argparse import ArgumentParser, Namespace

from mindspeed.features_manager.feature import MindSpeedFeature


class MuonFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('optimizer')

    def validate_args(self, args: Namespace):
        if args.optimizer == 'muon':
            if args.overlap_grad_reduce:
                raise ValueError("Muon optimizer does not support overlap grad reduce for now.")
            if args.overlap_param_gather:
                raise ValueError("Muon optimizer does not support overlap param gather for now.")
            if args.use_distributed_optimizer:
                raise ValueError("Muon optimizer does not support distributed optimizer for now.")
            if args.use_torch_fsdp2:
                raise ValueError("Muon optimizer does not support Torch-FSDP2 for now.")
            if args.ckpt_format not in ["torch", "torch_dist"]:
                raise ValueError("Muon optimizer supports torch and torch_dist checkpoint format.")

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        self.add_parser_argument_choices_value(parser, "--optimizer", 'muon')
        group.add_argument('--muon-momentum', type=float, default=0.9,
                        help='Momentum factor for Muon optimizer')
        group.add_argument('--muon-no-split-qkv', action='store_false', default=True,
                        dest='muon_split_qkv',
                        help='Whether to split QKV parameters for Muon optimizer')
        group.add_argument('--muon-use-nesterov', action='store_true',
                        help='Whether to use Nesterov-style momentum in the internal SGD')
        group.add_argument('--muon-scale-mode', type=str, default='spectral',
                        choices=['spectral', 'unit_rms_norm', 'shape_scaling'],
                        help='Scale mode for Muon optimizer')
        group.add_argument('--muon-fp32-matmul-prec', type=str, default='medium',
                        choices=['low', 'medium', 'high'],
                        help='FP32 matmul precision for Newton-Schulz iteration')
        group.add_argument('--muon-num-ns-steps', type=int, default=5,
                        help='Number of Newton-Schulz steps for Muon optimizer')
        group.add_argument('--muon-tp-mode', type=str, default='blockwise',
                        choices=['blockwise', 'duplicated', 'distributed'],
                        help='How to perform NS calculation for tensor model parallel weights')
        group.add_argument('--muon-extra-scale-factor', type=float, default=1.0,
                        help='Additional scale factor for the muon update')

    def register_patches(self, patch_manager, args):
        from mindspeed_llm.core.optimizer.muon import setup_model_and_optimizer_muon, muon_initialize_model_parallel_wrapper
        from mindspeed_llm.core.optimizer.optimizer_config import MuonOptimizerConfig

        if args.optimizer == 'muon':
            patch_manager.register_patch('megatron.core.optimizer.optimizer_config.OptimizerConfig', MuonOptimizerConfig)
            patch_manager.register_patch('megatron.training.training.setup_model_and_optimizer', setup_model_and_optimizer_muon)
            patch_manager.register_patch('megatron.core.parallel_state.initialize_model_parallel', muon_initialize_model_parallel_wrapper)