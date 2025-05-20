import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Union
from functools import wraps
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.training import get_args
from mindspeed_llm.tasks.models.ssm.state_space_duality import StateSpaceProcessor, ProcessInputs, StateOptions


def mamba_mixer_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        param_args = get_args()
        kwargs["rmsnorm"] = False
        kwargs["chunk_size"] = param_args.mamba_chunk_size
        kwargs["d_state"] = param_args.mamba_d_state
        kwargs["d_conv"] = param_args.mamba_d_conv
        kwargs["expand"] = param_args.mamba_expand
        kwargs["headdim"] = param_args.mamba_headdim    
        fn(self, *args, **kwargs)
        self.rmsnorm = True
        dt_min = kwargs.pop('dt_min', 0.001)
        dt_max = kwargs.pop('dt_max', 0.1)
        args = get_args()
        self.use_mem_eff_path = False
        self.d_ssm = param_args.mamba_d_ssm
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.d_ssm_local = self.d_inner_local if self.d_ssm is None else self.d_ssm // self.tensor_model_parallel_size


        if self.rmsnorm:
            self.norm = Mamba2RMSNorm(
                self.d_inner_local,
                eps=1e-5,
                group_size=self.d_inner_local // self.ngroups_local,
                norm_before_gate=self.norm_before_gate,
                device=torch.cuda.current_device(),
                dtype=self.config.params_dtype
            )
    return wrapper


def mamba_mixer_forward(self, hidden_states, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
    """
    hidden_states: (nL, B, D) / (L B D)
    Returns: same shape as hidden_states
    """
    seqlen_og = seqlen
    if seqlen is None:
        seqlen, batch, dim = hidden_states.shape
    else:
        batch_seqlen, dim = hidden_states.shape
        batch = batch_seqlen // seqlen

    conv_state, ssm_state = None, None
    if inference_params is not None:
        if self.config.sequence_parallel:
            raise AssertionError('sequence_parallel is unsupported in inference_params exist')
        conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
        if inference_params.seqlen_offset > 0:
            # The states are updated inplace
            out, out_bias, _, _ = self.step(hidden_states, conv_state, ssm_state)
            return out, out_bias

    A = -torch.exp(self.A_log.float())

    xz, _ = self.in_proj(hidden_states)

    # transpose: l b pd --> b l pd
    if seqlen_og is not None:
        xz = rearrange(xz, "(l b) d -> b l d", l=seqlen)
    else:
        xz = rearrange(xz, "l b d -> b l d").contiguous()

    d_mlp = (xz.shape[-1] - 2 * self.d_ssm_local - 2 * self.ngroups_local * self.d_state - self.nheads_local) // 2
    z0, x0, z, xBC, dt = torch.split(
        xz,
        [
            d_mlp,
            d_mlp,
            self.d_inner_local,
            self.d_inner_local + 2 * self.ngroups_local * self.d_state,
            self.nheads_local,
        ],
        dim=-1,
    )

    # transpose: b l pd --> b pd l
    xBC = rearrange(xBC, "b l d -> b d l").contiguous()

    # Compute short convolution
    if conv_state is not None:
        if cu_seqlens:
            raise('Variable length inputs in convolution are not currently supported')
        # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
        # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
        conv_state.copy_(
            F.pad(xBC, (self.d_conv - xBC.shape[-1], 0))
        )  # Update state (B D W)

    seqlen = xBC.size(2)
    if seq_idx:
        raise('Variable length inputs in convolution are not currently supported')    
    xBC = self.act(self.conv1d(xBC)[..., :seqlen])

    # transpose b pd l --> b l pd
    xBC = rearrange(xBC, "b d l ->  b l d").contiguous()

    x, B, C = torch.split(
        xBC,
        [
            self.d_inner_local,
            self.ngroups_local * self.d_state,
            self.ngroups_local * self.d_state,
        ],
        dim=-1,
    )

    config = {
        'nheads_local': self.nheads_local,
        'ngroups_local': self.ngroups_local,
        'dt_min': self.dt_min,
        'dt_max': self.dt_max,
        'dt_bias': self.dt_bias,        
        'headdim': self.headdim,
        'd_state': self.d_state,
        'chunk_size': self.chunk_size,
        'D_has_hdim': self.D_has_hdim
    }

    inputs = ProcessInputs(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=self.D
    )

    state_opts = StateOptions(
        return_final_state=True if ssm_state else False
    )
    state_space_duality = StateSpaceProcessor(config=config)
    y = state_space_duality.process(inputs, state_opts)          

    if ssm_state is not None:
        y, last_state, *rest = y
        if cu_seqlens is None:
            ssm_state.copy_(last_state)
        else:
            varlen_states = rest[0]
            ssm_state.copy_(varlen_states)

    if self.rmsnorm:
        y = rearrange(y, "b l h p -> b l (h p)").contiguous()
        y = self.norm(y, z=z)
    else:
        y = rearrange(y, "b l h p -> b l (h p)").contiguous()

    if d_mlp > 0:
        y = torch.cat([F.silu(z0) * x0, y], dim=-1) 

    if seqlen_og is not None:
        y = rearrange(y, "b l d -> (b l) d")          

    y = rearrange(y, "b l d -> l b d").contiguous()
    out, out_bias = self.out_proj(y)

    return out, out_bias 


class Mamba2RMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-5, group_size=None, norm_before_gate=True, device=None, dtype=None):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def _rms_norm_ref(self, x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True):
        dtype = torch.bfloat16
        N = x.shape[-1]
        weight = weight.float()
        bias = bias.float() if bias is not None else None
        if upcast:
            x = x.float()
            z = z.float() if z is not None else z
        if z is not None and not norm_before_gate:
            x = x * nn.functional.silu(z)
        if group_size is None:
            rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
            out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
        else:
            x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
            rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
            out = rearrange(x_group * rstd, "... g d -> ... (g d)") * weight
            if bias is not None:
                out = out + bias
        if z is not None and norm_before_gate:
            out *= nn.functional.silu(z)
        return out.to(dtype)        

    def forward(self, x, z=None):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))
        """
        return self._rms_norm_ref(x, self.weight, self.bias, z=z, eps=self.eps, group_size=self.group_size,
                            norm_before_gate=self.norm_before_gate)       