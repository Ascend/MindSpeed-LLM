#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import abc
import logging as logger
import os
from collections import defaultdict


logger.basicConfig(format="")
logger.getLogger().setLevel(logger.INFO)


class Convert(abc.ABC):
    def __init__(self, args):
        self.load_model = None
        self.save_model = None
        self.model_type_hf = args.model_type_hf
        self.transformer_impl = args.transformer_impl

        # parallel train arguments
        if not getattr(args, "enable_hf2mg_convert", False) and not getattr(args, "enable_mg2hf_convert", False):
            self.tensor_model_parallel_size = args.target_tensor_parallel_size
            self.pipeline_model_parallel_size = args.target_pipeline_parallel_size
            self.expert_model_parallel_size = args.target_expert_parallel_size
        self.expert_tensor_parallel_size = args.expert_tensor_parallel_size
        self.num_layer_list = args.num_layer_list
        self.noop_layers = args.noop_layers
        self.num_layers_per_virtual_pipeline_stage = args.num_layers_per_virtual_pipeline_stage
        self.pipeline_model_parallel_layout = getattr(args, "pipeline_model_parallel_layout", None)
        self.pipeline_layout = None
        self.virtual_pipeline_model_parallel_size = None
        self._validate_pipeline_distribution_args()

        # features arguments
        self.moe_grouped_gemm = args.moe_grouped_gemm
        self.moe_tp_extend_ep = args.moe_tp_extend_ep
        self.mla_mm_split = args.mla_mm_split
        self.schedules_method = args.schedules_method
        self.mtp_num_layers = 0 if args.mtp_num_layers is None else args.mtp_num_layers

        self.num_layers = args.num_layers
        self.first_k_dense_replace = args.first_k_dense_replace

    @staticmethod
    def _as_int_list(value):
        if value is None:
            return None
        if isinstance(value, str):
            return [int(item) for item in value.split(',')]
        return [int(item) for item in value]

    @staticmethod
    def _layout_as_text(value):
        if isinstance(value, str):
            return value
        input_data = getattr(value, "input_data", None)
        if isinstance(input_data, str):
            return input_data
        return repr(value)

    @staticmethod
    def _count_decoder_layers(stage):
        from mindspeed.core.pipeline_parallel.pipeline_model_parallel_layout.layout import LayerType

        return stage.count(LayerType.decoder)

    def _validate_pipeline_distribution_args(self):
        if self.num_layer_list is not None and self.pipeline_model_parallel_layout is not None:
            raise ValueError("num_layer_list and pipeline_model_parallel_layout are mutually exclusive")

    def initialize_pipeline_layout(self, num_layers, mtp_num_layers=0):
        """Parse and validate a flexible PP/VPP layout for offline conversion."""
        self._validate_pipeline_distribution_args()
        if self.pipeline_model_parallel_layout is None:
            return False
        if self.num_layers_per_virtual_pipeline_stage is not None:
            raise ValueError(
                "pipeline_model_parallel_layout and num_layers_per_virtual_pipeline_stage "
                "cannot be configured at the same time"
            )
        if self.schedules_method is not None:
            raise ValueError(
                "pipeline_model_parallel_layout and schedules_method cannot be configured at the same time"
            )
        if self.noop_layers is not None:
            raise ValueError("pipeline_model_parallel_layout and noop_layers cannot be configured at the same time")
        if mtp_num_layers:
            raise ValueError("pipeline_model_parallel_layout with MTP layers is not supported by checkpoint conversion")

        from mindspeed.core.pipeline_parallel.pipeline_model_parallel_layout.layout import (
            PipelineParallelLayerLayout,
        )

        layout = self.pipeline_model_parallel_layout
        if isinstance(layout, str):
            try:
                layout = PipelineParallelLayerLayout.from_str(layout, self.pipeline_model_parallel_size)
            except AssertionError as error:
                raise ValueError(str(error)) from error
        elif isinstance(layout, list):
            try:
                layout = PipelineParallelLayerLayout(layout, self.pipeline_model_parallel_size)
            except AssertionError as error:
                raise ValueError(str(error)) from error
        elif not isinstance(layout, PipelineParallelLayerLayout):
            raise TypeError(
                "pipeline_model_parallel_layout must be a string, list, or MindSpeed PipelineParallelLayerLayout, "
                f"but got {type(layout).__name__}"
            )

        try:
            layout.validate_layer_layout(num_layers=num_layers, mtp_num_layers=mtp_num_layers)
        except AssertionError as error:
            raise ValueError(str(error)) from error

        self.pipeline_layout = layout
        self.pipeline_model_parallel_layout = self._layout_as_text(layout)
        self.vpp_size = layout.virtual_pipeline_model_parallel_size
        self.virtual_pipeline_model_parallel_size = self.vpp_size if self.vpp_size > 1 else None
        return self.vpp_size > 1

    def get_layout_layer_mapping(self):
        """Return decoder layer IDs in Megatron's VPP-major, then PP-major order."""
        if self.pipeline_layout is None:
            raise RuntimeError("pipeline_model_parallel_layout has not been initialized")

        layer_mapping = defaultdict(dict)
        next_layer_id = 0
        for vpp_rank in range(self.vpp_size):
            for pp_rank in range(self.pipeline_model_parallel_size):
                num_decoder_layers = self._count_decoder_layers(self.pipeline_layout.layout[pp_rank][vpp_rank])
                layer_mapping[pp_rank][vpp_rank] = list(range(next_layer_id, next_layer_id + num_decoder_layers))
                next_layer_id += num_decoder_layers
        return layer_mapping

    def uses_virtual_pipeline(self):
        return getattr(self, "vpp_size", 1) > 1

    @staticmethod
    def mg_path_process(mg_path):
        """megatron model path"""
        iter_mg_path = os.path.join(mg_path, "iter_0000001")
        if not os.path.exists(mg_path):
            os.makedirs(mg_path, exist_ok=True)
        with open(os.path.join(mg_path, "latest_checkpointed_iteration.txt"), 'w', encoding='utf-8') as f:
            f.write("1")
        return iter_mg_path

    def generate_mg_weights_dir(self, tp_rank, pp_rank, ep_rank):
        """Generate the megatron weight directory."""
        if self.expert_model_parallel_size == 1 and self.pipeline_model_parallel_size == 1:
            prefix = f"mp_rank_{tp_rank:02}"
        elif self.expert_model_parallel_size == 1:
            prefix = f"mp_rank_{tp_rank:02}_{pp_rank:03}"
        elif self.pipeline_model_parallel_size == 1:
            prefix = f"mp_rank_{tp_rank:02}_{ep_rank:03}"
        else:
            prefix = f"mp_rank_{tp_rank:02}_{pp_rank:03}_{ep_rank:03}"
        return prefix

    def generate_pp_local_layer_idx(self):
        """generate each pp local layer index"""
        pp_local_layer_idx = defaultdict()

        if self.pipeline_layout is not None:
            for pp_rank in range(self.pipeline_model_parallel_size):
                num_decoder_layers = self._count_decoder_layers(self.pipeline_layout.layout[pp_rank][0])
                pp_local_layer_idx[pp_rank] = list(range(num_decoder_layers))
            return pp_local_layer_idx

        layer_list = self._as_int_list(self.num_layer_list)
        for pp_rank in range(self.pipeline_model_parallel_size):
            if layer_list is not None:
                pp_local_layer_idx[pp_rank] = list(range(layer_list[pp_rank]))
            else:
                pp_local_layer_idx[pp_rank] = list(range(self.num_layers // self.pipeline_model_parallel_size))

        if self.noop_layers is not None:
            noop_list = list(map(int, self.noop_layers.split(",")))
            num_layers_each_pp = self.num_layers // self.pipeline_model_parallel_size
            for num_noop_layers in noop_list:
                pp_idx = num_noop_layers // num_layers_each_pp
                local_noop_idx = num_noop_layers % num_layers_each_pp
                pp_local_layer_idx[pp_idx].remove(local_noop_idx)

        return pp_local_layer_idx

    def generate_vpp_local_layer_idx(self):
        vpp_local_layer_idx = defaultdict()
        for pp_rank in range(self.pipeline_model_parallel_size):
            vpp_local_layer_idx[pp_rank] = defaultdict()

        if self.pipeline_layout is not None:
            for pp_rank in range(self.pipeline_model_parallel_size):
                for vpp_rank in range(self.vpp_size):
                    num_decoder_layers = self._count_decoder_layers(self.pipeline_layout.layout[pp_rank][vpp_rank])
                    vpp_local_layer_idx[pp_rank][vpp_rank] = list(range(num_decoder_layers))
            return vpp_local_layer_idx

        for pp_rank in range(self.pipeline_model_parallel_size):
            for vpp_rank in range(self.vpp_size):
                vpp_local_layer_idx[pp_rank][vpp_rank] = list(range(self.num_layers_per_virtual_pipeline_stage))

        if self.noop_layers is not None:
            noop_list = list(map(int, self.noop_layers.split(",")))
            num_layers_each_pp = self.num_layers // self.pipeline_model_parallel_size

            if self.schedules_method == 'dualpipev':
                # calc pp rank, vpp rank and local idx of noop layer
                for noop_layer in noop_list:
                    # e.g. pp2 noop5 [0 1 6 7 | 2 3 4 5] -> layer5: pp1 vpp1 local_idx1
                    # layer5 and layer2 are symmetrical, so they are in the same pp_rank.
                    # all layer are divided into two parts. layer5 is in last part. so vpp_rank=1
                    if noop_layer >= self.num_layers // 2:
                        mapping_layer = -(noop_layer - self.num_layers + 1)
                        vpp_idx = 1
                        pp_idx = mapping_layer // ((self.num_layers // 2) // self.pipeline_model_parallel_size)
                        local_noop_idx = (
                            self.num_layers_per_virtual_pipeline_stage
                            - 1
                            - (mapping_layer - pp_idx * self.num_layers_per_virtual_pipeline_stage)
                        )
                    else:
                        vpp_idx = 0
                        pp_idx = noop_layer // ((self.num_layers // 2) // self.pipeline_model_parallel_size)
                        local_noop_idx = noop_layer - pp_idx * self.num_layers_per_virtual_pipeline_stage
                    vpp_local_layer_idx[pp_idx][vpp_idx].remove(local_noop_idx)
            else:
                for num_noop_layer in noop_list:
                    pp_idx = (
                        num_noop_layer
                        % (self.pipeline_model_parallel_size * self.num_layers_per_virtual_pipeline_stage)
                        // self.num_layers_per_virtual_pipeline_stage
                    )
                    vpp_idx = (
                        num_noop_layer
                        // self.num_layers_per_virtual_pipeline_stage
                        // self.pipeline_model_parallel_size
                    )
                    local_noop_idx = num_noop_layer % num_layers_each_pp % self.num_layers_per_virtual_pipeline_stage
                    vpp_local_layer_idx[pp_idx][vpp_idx].remove(local_noop_idx)

        return vpp_local_layer_idx

    @abc.abstractmethod
    def set_model_preprocess(self, weights_dict, mg_model):
        """Embedding layer process"""
        pass

    @abc.abstractmethod
    def set_model_postprocess(self, weights_dict, mg_model):
        """Final norm & LM Head process"""
        pass

    @abc.abstractmethod
    def set_model_layer_norm(self, hf_layer_idx, local_layer_idx, weights_dict, mg_model, mtp_layer_flag=False):
        """Layernorm process"""
        pass

    @abc.abstractmethod
    def set_model_layer_attn(self, hf_layer, local_layer_idx, weights_dict, mg_model, mtp_layer_flag=False):
        """Attention layer process"""
        pass

    @abc.abstractmethod
    def set_model_layer_mlp(self, hf_layer_idx, local_layer_idx, weights_dict, mg_model, mtp_layer_flag=False):
        """MLP layer process"""
        pass
