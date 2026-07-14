# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Ascend fused MHC operator wrappers.

This module keeps the model-side MHC implementation independent from the
operator package. It adapts tensor layouts between the model convention
([s, b, ...]) and the Ascend fused operator convention ([b, s, ...]).
"""

import torch


def _mhc_ops():
    """Load Ascend MHC fused operators only when the feature is enabled."""
    try:
        import cann_ops_transformer
    except ImportError as exc:
        raise ImportError("cann_ops_transformer is required when --use-ascend-mhc is enabled.") from exc
    return cann_ops_transformer.ops


def mhc_pre_sinkhorn_ascend(x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps):
    """Run the Ascend fused MHC pre-sinkhorn operator.

    Args:
        x: Model-side hidden states in [s, b, hc, d].
        phi: Projection weight from hc_fn.
        alpha: MHC scale parameter.
        bias: MHC base parameter.
        hc_mult: Number of MHC heads.
        num_iters: Sinkhorn iteration count.
        hc_eps: Epsilon used by the MHC sinkhorn computation.
        norm_eps: Epsilon used by RMS normalization.

    Returns:
        A tuple of model-side tensors: h_in [s, b, d], h_post [s, b, hc],
        and h_res [s, b, hc, hc].
    """
    ops = _mhc_ops()
    # The model path keeps MHC tensors in [s, b, ...], while the Ascend op uses [b, s, ...].
    x_bsnd = x.permute(1, 0, 2, 3).contiguous()

    outputs = ops.mhc_pre_sinkhorn(
        x_bsnd,
        phi,
        alpha,
        bias,
        hc_mult,
        num_iters,
        hc_eps,
        norm_eps,
    )
    h_in, h_post, h_res = outputs[:3]

    batch_size, seq_length = x_bsnd.shape[:2]
    h_in = h_in.view(batch_size, seq_length, h_in.shape[-1]).permute(1, 0, 2).contiguous()
    h_post = h_post.view(batch_size, seq_length, h_post.shape[-1]).permute(1, 0, 2).contiguous()
    h_res = h_res.view(batch_size, seq_length, hc_mult, hc_mult).permute(1, 0, 2, 3).contiguous()
    return h_in, h_post, h_res


def mhc_post_ascend(h_out, x, h_post, h_res):
    """Run the Ascend fused MHC post operator.

    Args:
        h_out: Current hidden states in [s, b, d].
        x: Residual MHC states in [s, b, hc, d].
        h_post: Post weights in [s, b, hc].
        h_res: Residual mixing weights in [s, b, hc, hc].

    Returns:
        Model-side fused post output in [s, b, hc, d].
    """
    ops = _mhc_ops()
    h_out_bsd = h_out.permute(1, 0, 2).contiguous()
    x_bsnd = x.permute(1, 0, 2, 3).contiguous()
    h_post_bsn = h_post.permute(1, 0, 2).contiguous()
    h_res_bsnn = h_res.permute(1, 0, 2, 3).contiguous()

    out = ops.mhc_post(x_bsnd, h_res_bsnn, h_out_bsd, h_post_bsn)
    # Pipeline deallocation requires its output not to be a view. `permute`
    # creates a view, so materialize a contiguous tensor before returning it.
    return out.permute(1, 0, 2, 3).clone(memory_format=torch.contiguous_format)
