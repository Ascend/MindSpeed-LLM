# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from dataclasses import dataclass
from typing import Optional, Union

import torch

from mindspeed_llm.tasks.models.transformer.deepseek4.mhc.pre_bmm import hc_pre_bmm_forward

from megatron.core.transformer import ModuleSpec, TransformerLayer, TransformerLayerSubmodules


def copy_untyped_storage(dst: torch.Tensor, src: torch.Tensor):
    dst.untyped_storage().resize_(src.untyped_storage().size())
    dst.untyped_storage().copy_(src.untyped_storage())
    src.untyped_storage().resize_(0)


@dataclass
class MHCRecomputeInfo:
    layer: TransformerLayer = None
    # For MHC 
    num_stream: Optional[int] = None
    is_mtp: Optional[bool] = False
    hc_pre_input: Optional[torch.Tensor] = None
    hc_pre_input_fp32: Optional[torch.Tensor] = None
    
    h_pre: Optional[torch.Tensor] = None
    h_pre_out: Optional[torch.Tensor] = None

    attention_output_with_bias: Optional[torch.Tensor] = None
    residual: Optional[torch.Tensor] = None
    h_post: Optional[torch.Tensor] = None
    h_res: Optional[torch.Tensor] = None
    hc_post_out: Optional[torch.Tensor] = None

    hc_pre_input_mlp: Optional[torch.Tensor] = None
    mlp_hc_pre_input_fp32: Optional[torch.Tensor] = None

    mlp_h_pre: Optional[torch.Tensor] = None
    mlp_h_pre_out: Optional[torch.Tensor] = None

    mlp_residual: Optional[torch.Tensor] = None
    mlp_h_post: Optional[torch.Tensor] = None
    mlp_h_res: Optional[torch.Tensor] = None
    mlp_output_with_bias: Optional[torch.Tensor] = None
    mlp_hc_post_out: Optional[torch.Tensor] = None

    use_mhc_triton: Optional[bool] = False
    is_last_layer: Optional[bool] = False

    def drop_mhc_pre_attn(self):
        self.hc_pre_input_fp32.untyped_storage().resize_(0)
        self.h_pre_out.untyped_storage().resize_(0)

    def recompute_pre_attn(self, layer_graph=None):
        if layer_graph is not None:
            self.layer = layer_graph.layer
        
        with torch.no_grad():
            copy_untyped_storage(self.hc_pre_input_fp32, self.hc_pre_input.float())
            if self.use_mhc_triton:
                y = hc_pre_bmm_forward(self.h_pre, 
                    self.hc_pre_input_fp32.unflatten(dim=-1, sizes=(self.layer.config.hc_mult, -1)))

            y = y.to(self.hc_pre_input.dtype)
            copy_untyped_storage(self.h_pre_out, y)

    def drop_mhc_pre_mlp(self):
        self.mlp_hc_pre_input_fp32.untyped_storage().resize_(0)
        self.mlp_h_pre_out.untyped_storage().resize_(0)

    def recompute_pre_mlp(self, layer_graph=None):
        if layer_graph is not None:
            self.layer = layer_graph.layer
        with torch.no_grad():
            if self.use_mhc_triton:
                mlp_h_pre_out_y = hc_pre_bmm_forward(self.mlp_h_pre, 
                    self.mlp_hc_pre_input_fp32.unflatten(dim=-1, sizes=(self.layer.config.hc_mult, -1)))

            mlp_h_pre_out_y = mlp_h_pre_out_y.to(self.hc_post_out.dtype)
            copy_untyped_storage(self.mlp_h_pre_out, mlp_h_pre_out_y)

    def drop_mhc_post_attn(self):
        self.hc_post_out.untyped_storage().resize_(0)

    def recompute_post_attn(self, layer_graph=None):
        if layer_graph is not None:
            self.layer = layer_graph.layer

        with torch.no_grad():

            hc_post_out = self.layer.attn_mhc(self.attention_output_with_bias,
                                              mhc_stage='post', 
                                              residual=self.residual,
                                              post=self.h_post,
                                              comb=self.h_res)

            mlp_hc_pre_input_fp32 = hc_post_out.float()
            copy_untyped_storage(self.mlp_hc_pre_input_fp32, mlp_hc_pre_input_fp32)
            copy_untyped_storage(self.hc_post_out, hc_post_out)

    def drop_mhc_post_mlp(self):
        self.mlp_hc_post_out.untyped_storage().resize_(0)

    def recompute_post_mlp(self, layer_graph=None):
        if layer_graph is not None:
            self.layer = layer_graph.layer
        with torch.no_grad():
            mlp_hc_post_out = self.layer.mlp_mhc(self.mlp_output_with_bias,
                                                 mhc_stage='post', 
                                                 residual=self.mlp_residual,
                                                 post=self.mlp_h_post,
                                                 comb=self.mlp_h_res)

            copy_untyped_storage(self.mlp_hc_post_out, mlp_hc_post_out)


    def drop(self):
        self.drop_mhc_pre_attn()
        self.drop_mhc_post_attn()
        self.drop_mhc_pre_mlp()
        # if not self.is_last_layer:
        #     self.drop_mhc_post_mlp()

    def recompute(self):
        self.recompute_pre_attn()
        self.recompute_post_attn()
        self.recompute_pre_mlp()
        # if not self.is_last_layer:
        #     self.recompute_post_mlp()

    def detach(self):
        for attr in vars(self):
            setattr(self, attr, None)


class RecomputeInputWrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, recompute_info: MHCRecomputeInfo):
        ctx.recompute_info = recompute_info
        return x

    @staticmethod
    def backward(ctx, grad_x):
        recompute_info: MHCRecomputeInfo = ctx.recompute_info
        recompute_info.detach()
        return grad_x, None


class RecomputeOutputWrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, recompute_info: MHCRecomputeInfo):
        ctx.recompute_info = recompute_info
        recompute_info.drop()
        return x

    @staticmethod
    def backward(ctx, grad_x):
        recompute_info: MHCRecomputeInfo = ctx.recompute_info
        recompute_info.recompute()
        return grad_x, None