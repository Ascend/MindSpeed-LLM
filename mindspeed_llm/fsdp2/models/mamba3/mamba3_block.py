# Copyright (c) 2026, Dao AI Lab, Goombalab.
# This code is inspired by the state-spaces/mamba library.


import math
from typing import Optional, Tuple

from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mindspeed_llm.fsdp2.utils.global_vars import get_args


def rms_norm_ref(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True):
    dtype = x.dtype
    N = x.shape[-1]
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
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
        out *= F.silu(z)
    return out.to(dtype)


class RMSNorm(torch.nn.Module):
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

    def forward(self, x, z=None):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))
        """
        return rms_norm_ref(x, self.weight, self.bias, z=z, eps=self.eps, group_size=self.group_size,
                    norm_before_gate=self.norm_before_gate)


class Mamba3(nn.Module):
    def __init__(
    self,
    d_model,
    d_state=128,
    expand=2,
    headdim=64,
    ngroups=1,
    # ----------------------------------------
    # Mamba-3 configs
    rope_fraction=0.5,
    dt_min=0.001,
    dt_max=0.1,
    dt_init_floor=1e-4,
    A_floor=1e-4,
    is_outproj_norm=False,
    is_mimo=False,
    mimo_rank=4,
    #-------------------------------------------
    # Fused kernel and sharding options
    chunk_size=64, # Recommended: 64 for SISO, 64/mimo_rank for MIMO
    dropout=0.0, # Just to absorb the kwarg
    layer_idx=None, # Absorb kwarg for general module
    n_layer=None, # Absorb kwarg for general module
    device=None,
    dtype=None,
    **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.A_floor = A_floor
        self.is_outproj_norm = is_outproj_norm
        self.is_mimo = is_mimo
        self.mimo_rank = mimo_rank
        if not self.is_mimo:
            self.mimo_rank = 1

        self.dt_max = dt_max
        self.dt_min = dt_min
        self.dt_init_floor = dt_init_floor

        self.d_inner = int(self.expand * self.d_model)
        if self.d_inner % self.headdim != 0:
            raise ValueError(
                f"Dimension mismatch: d_inner={self.d_inner} must be divisible by headdim={self.headdim}"
            )
        self.nheads = self.d_inner // self.headdim
        self.num_bc_heads = ngroups
        
        # RoPE flags
        if rope_fraction not in (0.5, 1.0):
            raise ValueError(f"rope_fraction must be either 0.5 or 1.0, got {rope_fraction}")
        self.rotary_dim_divisor = int(2 / rope_fraction)
        self.split_tensor_size = int(d_state * rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2
        if self.num_rope_angles <= 0:
            raise ValueError(f"num_rope_angles must be greater than 0, got {self.num_rope_angles}")

        # Order: [z, x, B, C, dd_dt, dd_A, trap, angle]
        d_in_proj = 2 * self.d_inner + 2 * self.d_state * self.num_bc_heads * self.mimo_rank + 3 * self.nheads + self.num_rope_angles
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)

        # dt_bias parameterization        
        _dt = torch.exp(
            torch.rand(self.nheads, device=device, dtype=torch.float32) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        _dt = torch.clamp(_dt, min=dt_init_floor)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias.to(device=device), requires_grad=True)
        self.dt_bias._no_weight_decay = True

        # B and C biases
        self.B_bias = nn.Parameter(1 + torch.zeros((self.nheads, self.mimo_rank, self.d_state), dtype=torch.float32, device=device), requires_grad=True)
        self.C_bias = nn.Parameter(1 + torch.zeros((self.nheads, self.mimo_rank, self.d_state), dtype=torch.float32, device=device), requires_grad=True)

        # RMS Norm for B and C
        args = get_args()
        self.use_triton_rmsnormgated = args.use_triton_rmsnormgated
        if self.use_triton_rmsnormgated:
            from mindspeed_llm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
            if RMSNormGated is None:
                raise ValueError("RMSNormGated cannot be None")
            self.B_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)
            self.C_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)
        else:
            self.B_norm = RMSNorm(self.d_state, eps=1e-5, **factory_kwargs)
            self.C_norm = RMSNorm(self.d_state, eps=1e-5, **factory_kwargs)

        if self.is_mimo:
            # Initialize up/down MIMO projection (for x and z)
            mimo_x_init_weights = torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device) / self.mimo_rank
            mimo_z_init_weights = torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device)
            mimo_o_init_weights = torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device) / self.mimo_rank

            self.mimo_x = nn.Parameter(mimo_x_init_weights, requires_grad=True)
            self.mimo_z = nn.Parameter(mimo_z_init_weights, requires_grad=True)
            self.mimo_o = nn.Parameter(mimo_o_init_weights, requires_grad=True)

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.is_outproj_norm:
            if self.use_triton_rmsnormgated:
                self.norm = RMSNormGated(
                    self.d_inner,
                    eps=1e-5,
                    norm_before_gate=True,
                    group_size=self.headdim,
                    **factory_kwargs
                )
            else:
                self.norm = RMSNorm(
                    self.d_inner,
                    eps=1e-5,
                    norm_before_gate=True,
                    group_size=self.headdim,
                    **factory_kwargs
                )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)



    def forward(self, u, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        u: (batch, seqlen, hidden_dim)
        Returns: same shape as u
        """

        batch, seqlen, dim = u.shape
        if cu_seqlens is not None:
            raise NotImplementedError("Currently does not support varlen in Mamba-3 (MIMO).")

        angle_dt_state, ssm_state, k_state, v_state = None, None, None, None


        # Apply in_proj

        zxBCdtAtrap = self.in_proj(u)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdtAtrap,
            [
                self.d_inner, self.d_inner, 
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.nheads, self.nheads, self.nheads, 
                self.num_rope_angles
            ],
            dim=-1)
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
        C = rearrange(C, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
        trap = rearrange(trap, "b l h -> b h l")

        # Compute ADT, DT
        _A = -F.softplus(dd_A.to(torch.float32)) # (B, L, N)
        _A = torch.clamp(_A, max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias) # (B, L, N)

        ADT = _A * DT
        DT = rearrange(DT, "b l n -> b n l")
        ADT = rearrange(ADT, "b l n -> b n l")

        # Compute angle
        angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads, -1) # (B, L, N, S)

        # Apply RMS Norm on B and C
        B = self.B_norm(B)
        C = self.C_norm(C)

        # Apply Mamba-3 kernel

        if self.is_mimo:
            angles = apply_angle_dt_reference(angles, DT.transpose(-1, -2))
            DA_CS, DA_CS_REV, Segsum = dacs_segsum_torch(ADT, self.chunk_size)
            Out, Final_SSM_State, Final_K = mamba3_MIMO_chunk_ref(
                q=C,
                k=B,
                v=x,
                dA_cs=DA_CS,
                dA_cs_rev=DA_CS_REV,
                dt=DT,
                trap=trap,
                q_bias=self.C_bias,
                k_bias=self.B_bias,
                mimo_v=self.mimo_x,
                mimo_z=self.mimo_z,
                mimo_o=self.mimo_o if not self.is_outproj_norm else None,
                angles=angles,
                D=self.D,
                z=z if not self.is_outproj_norm else None,
                chunk_size=self.chunk_size,
                rotary_dim_divisor=self.rotary_dim_divisor,
                dtype=x.dtype,
                return_final_state=ssm_state is not None,
            )

            y = rearrange(Out, "b l h p -> b l (h p)")
        else:
            y, states_ref = mamba3_siso_fwd_ref(
                Q=C.squeeze(2),
                K=B.squeeze(2),
                V=x,
                ADT=ADT,
                DT=DT,
                Trap=trap,
                Q_bias=self.C_bias.squeeze(1),
                K_bias=self.B_bias.squeeze(1),
                Angles=angles,
                D=self.D,
                Z=z if not self.is_outproj_norm else None,
                chunk_size=self.chunk_size,
                Initial_States=None,

                )

            if ssm_state is not None:
                y, last_angle, last_state, last_k, last_v, *rest = y
                angle_dt_state.copy_(last_angle)
                ssm_state.copy_(last_state)
                k_state.copy_(last_k)
                v_state.copy_(last_v)
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.is_outproj_norm:
                z = rearrange(z, "b l h p -> b l (h p)")
                y = self.norm(y, z)
        
        out = self.out_proj(y.to(x.dtype))
        return out


    def _preprocess(self, A_proj, dd_dt, B, C, x, z, trap_proj, angle_proj):
        _A = -F.softplus(A_proj.to(torch.float32))
        _A = torch.clamp(_A, max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        trap = torch.sigmoid(trap_proj)

        rank = self.mimo_rank if self.is_mimo else 1
        B = rearrange(B, "b (r g s) -> b r g s", g=self.num_bc_heads, r=rank)
        C = rearrange(C, "b (r g s) -> b r g s", g=self.num_bc_heads, r=rank)

        B = self.B_norm(B)
        C = self.C_norm(C)

        B = B.expand(-1, -1, self.nheads, -1) # (B, R, N, S)
        C = C.expand(-1, -1, self.nheads, -1) # (B, R, N, S)

        x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
        z = rearrange(z, "b (h p) -> b h p", p=self.headdim)

        angles = angle_proj.unsqueeze(-2).expand(-1, self.nheads, -1)

        return DT, B, C, x, z, trap, _A, angles


def _segsum(x: torch.Tensor) -> torch.Tensor:
    """Segment sum helper for attention computation."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def mamba3_siso_fwd_ref(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    ADT: torch.Tensor,
    DT: torch.Tensor,
    Trap: torch.Tensor,
    Q_bias: torch.Tensor,
    K_bias: torch.Tensor,
    Angles: torch.Tensor,
    D,
    Z,
    Initial_States=None,
    chunk_size: int = 64,
    dtype: torch.dtype = torch.float32,
    cu_seqlens=None,
    ):
    """
    Args:
        Initial_States: Optional tuple of (Angle_State, SSM_State, K_State, V_State)

    Returns:
        out_z: Output with Z gating applied
        final_states: (Final_Angle_State, Final_SSM_State, Final_K_State, Final_V_State)
    """
    batch, total_seqlen, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape
    headdim_angles = Angles.shape[-1]
    device = Q.device

    is_varlen = cu_seqlens is not None
    if is_varlen and batch != 1:
        raise ValueError(f"batch must be 1 when is_varlen is True, got batch={batch}")

    # Cast inputs
    Q = Q.to(dtype)
    K = K.to(dtype)
    V = V.to(dtype)
    ADT = ADT.to(torch.float32)
    DT = DT.to(torch.float32)
    Trap = Trap.to(dtype)
    Q_bias = Q_bias.to(dtype)
    K_bias = K_bias.to(dtype)
    Angles = Angles.to(dtype)
    if D is not None:
        D = D.to(dtype)
    if Z is not None:
        Z = Z.to(dtype)
    if Initial_States is not None:
        Initial_Angle_State, Initial_SSM_State, Initial_K_State, Initial_V_State = Initial_States

    Angles = torch.tanh(Angles) * math.pi
    # Expand Q/K for GQA
    if Q.shape[2] != V.shape[2]:
        Q = repeat(Q, "b s h_bc d -> b s (h_bc g) d", g=V.shape[2] // Q.shape[2])
    if K.shape[2] != V.shape[2]:
        K = repeat(K, "b s h_bc d -> b s (h_bc g) d", g=V.shape[2] // K.shape[2])

    out_zs = []
    Final_Angle_States = []
    Final_SSM_States = []
    Final_K_States = []
    Final_V_States = []

    TWO_PI = 2 * math.pi

    def _rotary(tensor, cos, sin):
        tensor_reshaped = tensor.view(*tensor.shape[:-1], -1, 2)
        tensor_0 = tensor_reshaped[..., 0]
        tensor_1 = tensor_reshaped[..., 1]
        if cos.shape[-1] < tensor_0.shape[-1]:
            pad_size = tensor_0.shape[-1] - cos.shape[-1]
            cos = F.pad(cos, (0, pad_size), value=1.0)
            sin = F.pad(sin, (0, pad_size), value=0.0)
        rotated_0 = tensor_0 * cos - tensor_1 * sin
        rotated_1 = tensor_0 * sin + tensor_1 * cos
        return torch.stack([rotated_0, rotated_1], dim=-1).view_as(tensor)

    def compute_one_sequence(seq_idx):
        if is_varlen:
            start_idx, end_idx = cu_seqlens[seq_idx].item(), cu_seqlens[seq_idx + 1].item()
            Q_curr = Q[0, start_idx:end_idx, :, :]
            K_curr = K[0, start_idx:end_idx, :, :]
            V_curr = V[0, start_idx:end_idx, :, :]
            ADT_curr = ADT[0, :, start_idx:end_idx]
            DT_curr = DT[0, :, start_idx:end_idx]
            Trap_curr = Trap[0, :, start_idx:end_idx]
            Angles_curr = Angles[0, start_idx:end_idx, :, :]
            Z_curr = Z[0, start_idx:end_idx, :, :] if Z is not None else None
        else:
            Q_curr = Q[seq_idx]
            K_curr = K[seq_idx]
            V_curr = V[seq_idx]
            ADT_curr = ADT[seq_idx]
            DT_curr = DT[seq_idx]
            Trap_curr = Trap[seq_idx]
            Angles_curr = Angles[seq_idx]
            Z_curr = Z[seq_idx] if Z is not None else None

        Trap_curr = torch.sigmoid(Trap_curr)
        seqlen_curr = Q_curr.shape[0]

        Angles_scaled = Angles_curr.float() * DT_curr.transpose(0, 1).unsqueeze(-1)
        Angles_Cumsum = torch.cumsum(Angles_scaled, dim=0)
        if Initial_States is not None:
            Initial_Angle_State_curr = Initial_Angle_State[seq_idx]
            Angles_Cumsum = Angles_Cumsum + Initial_Angle_State_curr.unsqueeze(0)
        Angles_Cumsum = Angles_Cumsum - TWO_PI * torch.floor(Angles_Cumsum / TWO_PI)
        Final_Angle_States.append(Angles_Cumsum[-1])

        # Initialize acc_states
        if Initial_States is not None:
            Initial_SSM_State_curr = Initial_SSM_State[seq_idx]
            Initial_K_State_curr = Initial_K_State[seq_idx]
            Initial_V_State_curr = Initial_V_State[seq_idx]

            scalar = DT_curr[:, 0] * (1 - Trap_curr[:, 0])
            acc_states = Initial_SSM_State_curr + Initial_V_State_curr[:, :, None] * Initial_K_State_curr[:, None, :] * scalar[:, None, None]
        else:
            acc_states = torch.zeros((nheads, headdim_v, headdim_qk), device=device, dtype=torch.float32)

        # Compute shifted gamma and scale
        DT_shifted = F.pad(DT_curr[:, 1:], (0, 1))
        Trap_shifted = F.pad(Trap_curr[:, 1:], (0, 1))
        shifted_gamma = DT_shifted * (1 - Trap_shifted)
        scale = DT_curr * Trap_curr + DT_shifted * (1 - Trap_shifted)

        # Add biases
        Q_curr = Q_curr + Q_bias.unsqueeze(0)
        K_curr = K_curr + K_bias.unsqueeze(0)

        # Compute QK dot for skip connection
        QK_dot = torch.sum(K_curr * Q_curr, dim=-1) * shifted_gamma.transpose(0, 1)

        # Rotary embeddings using Angles_Cumsum
        cos_angles_curr = torch.cos(Angles_Cumsum).to(Q_curr.dtype)
        sin_angles_curr = torch.sin(Angles_Cumsum).to(Q_curr.dtype)
        Q_curr = _rotary(Q_curr, cos_angles_curr, sin_angles_curr)
        K_curr = _rotary(K_curr, cos_angles_curr, sin_angles_curr)

        Final_K_States.append(K_curr[-1])
        Final_V_States.append(V_curr[-1])

        K_curr_scaled = K_curr * scale.transpose(0, 1).unsqueeze(-1).to(K_curr.dtype)
        
        # Compute output via quadratic attention
        QK = torch.einsum("thd,shd->hts", Q_curr, K_curr_scaled)
        QK_causal = torch.tril(QK)
        QK_causal = (QK_causal * torch.exp(_segsum(ADT_curr))).to(QK_causal.dtype)
        out = torch.einsum("hts,shd->thd", QK_causal, V_curr)

        if Initial_States is not None:
            da_cs = torch.cumsum(ADT_curr, dim=-1)
            exp_da_cs = torch.exp(da_cs)
            out = out + torch.einsum("hDd,thd,ht->thD", acc_states.to(Q_curr.dtype), Q_curr, exp_da_cs.to(Q_curr.dtype))

        if D is not None:
            out = out + D[None, :, None] * V_curr
        out = out - V_curr * QK_dot.unsqueeze(-1)

        if Z_curr is not None:
            out = out * Z_curr * torch.sigmoid(Z_curr)
        out_zs.append(out)

        # Compute final state
        da_cs_last = torch.exp(torch.sum(ADT_curr, dim=-1))
        da_cs_rev = torch.exp(torch.sum(ADT_curr, dim=-1, keepdim=True) - torch.cumsum(ADT_curr, dim=-1))
        V_curr_scaled = V_curr * da_cs_rev.permute(1, 0).unsqueeze(-1).to(V_curr.dtype)
        final_acc_states = acc_states * da_cs_last.unsqueeze(-1).unsqueeze(-1) + torch.einsum(
            "thd,thD->hDd", K_curr_scaled, V_curr_scaled.to(K_curr_scaled.dtype))
        Final_SSM_States.append(final_acc_states)


    num_sequences = cu_seqlens.size(0) - 1 if is_varlen else batch
    for seq_idx in range(num_sequences):
        compute_one_sequence(seq_idx)

    if not is_varlen:
        out_zs = torch.stack(out_zs, dim=0)
        Final_Angle_States = torch.stack(Final_Angle_States, dim=0)
        Final_SSM_States = torch.stack(Final_SSM_States, dim=0)
        Final_K_States = torch.stack(Final_K_States, dim=0)
        Final_V_States = torch.stack(Final_V_States, dim=0)
    else:
        out_zs = torch.cat(out_zs, dim=0).unsqueeze(0)
        Final_Angle_States = torch.stack(Final_Angle_States, dim=0)
        Final_SSM_States = torch.stack(Final_SSM_States, dim=0)
        Final_K_States = torch.stack(Final_K_States, dim=0)
        Final_V_States = torch.stack(Final_V_States, dim=0)

    return out_zs, (Final_Angle_States, Final_SSM_States, Final_K_States, Final_V_States)


def mamba3_MIMO_chunk_ref(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_bias: Tensor,
    k_bias: Tensor,
    mimo_v: Tensor,
    mimo_o: Optional[Tensor],
    z: Optional[Tensor],
    mimo_z: Optional[Tensor],
    angles: Tensor,
    dA_cs: Tensor,
    dA_cs_rev: Tensor,
    dt: Tensor,
    trap: Tensor,
    D: Optional[Tensor],
    chunk_size: int = 64,
    rotary_dim_divisor: int = 4,
    return_final_state: bool = False,
    dtype: torch.dtype = torch.float32,
    rotate_pairwise: bool = False,
    contract_mimo_out: bool = True,
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:

    nchunks = q.shape[1] // chunk_size
    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    if z is not None:
        z = z.to(dtype)
        mimo_z = mimo_z.to(dtype)
    if D is not None:
        D = D.to(dtype)
    q_bias, k_bias = q_bias.to(dtype), k_bias.to(dtype)
    mimo_v = mimo_v.to(dtype)
    if contract_mimo_out:
        if mimo_o is None:
            raise ValueError("mimo_o cannot be None when contract_mimo_out is True")
        mimo_o = mimo_o.to(dtype)
    if dA_cs is not None:
        dA_cs, dA_cs_rev = dA_cs.to(dtype), dA_cs_rev.to(dtype)
        dA_cs = rearrange(dA_cs, "b h (n c) -> b h n c", c=chunk_size)
        dA_cs_rev = rearrange(dA_cs_rev, "b h (n c) -> b h n c", c=chunk_size)

    batch, seqlen, mimo_rank, nheads_qk, dstate = q.shape
    nheads = v.shape[-2]
    if nheads_qk != nheads:
        q = repeat(q, "b s r h_qk d -> b s r (h_qk g) d", g=nheads // nheads_qk)
        k = repeat(k, "b s r h_qk d -> b s r (h_qk g) d", g=nheads // nheads_qk)

    angles = angles.to(dtype) if angles is not None else None
    trap = trap.to(dtype) if trap is not None else None
    dt = dt.to(dtype) if dt is not None else None

    q_bias = rearrange(q_bias, "h r d -> r h d")
    k_bias = rearrange(k_bias, "h r d -> r h d")
    q = q + q_bias[None, None, :, :, :]
    k = k + k_bias[None, None, :, :, :]

    qk_dot = torch.einsum("bsRhd,bsrhd->bsRrh", q, k)

    if angles is not None:
        angles = angles.unsqueeze(2)
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        def apply_rotary_emb(tensor: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
            if rotate_pairwise:
                tensor_reshaped = tensor.view(*tensor.shape[:-1], -1, 2)
                tensor_0 = tensor_reshaped[..., 0]
                tensor_1 = tensor_reshaped[..., 1]
                rotated_0 = tensor_0 * cos - tensor_1 * sin
                rotated_1 = tensor_0 * sin + tensor_1 * cos
                return torch.stack([rotated_0, rotated_1], dim=-1).view_as(tensor)
            # Kernel-aligned convention.
            tensor_reshaped = tensor.view(*tensor.shape[:-1], 2, -1)
            tensor_0 = tensor_reshaped[..., 0, :]
            tensor_1 = tensor_reshaped[..., 1, :]
            rotated_0 = tensor_0 * cos - tensor_1 * sin
            rotated_1 = tensor_0 * sin + tensor_1 * cos
            return torch.stack([rotated_0, rotated_1], dim=-2).view_as(tensor)

        def apply_rotary_emb_rotate_half(tensor: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
            tensor_reshaped = tensor.view(*tensor.shape[:-1], 4, -1)
            tensor_0 = tensor_reshaped[..., 0, :]
            tensor_1 = tensor_reshaped[..., 2, :]
            rotated_0 = tensor_0 * cos - tensor_1 * sin
            rotated_1 = tensor_0 * sin + tensor_1 * cos
            return torch.stack(
                [
                    rotated_0,
                    tensor_reshaped[..., 1, :],
                    rotated_1,
                    tensor_reshaped[..., 3, :],
                ],
                dim=-2,
            ).view_as(tensor)

        if rotary_dim_divisor == 4:
            q = apply_rotary_emb_rotate_half(q, cos_angles, sin_angles)
            k = apply_rotary_emb_rotate_half(k, cos_angles, sin_angles)
        elif rotary_dim_divisor == 2:
            q = apply_rotary_emb(q, cos_angles, sin_angles)
            k = apply_rotary_emb(k, cos_angles, sin_angles)
        else:
            raise ValueError(f"Invalid rotary_dim_divisor: {rotary_dim_divisor}")

    if return_final_state:
        final_k = k[:, -1].contiguous().clone()
    else:
        final_k = None

    trap = torch.nn.functional.sigmoid(trap)
    gamma = dt * trap
    dt_shifted = torch.nn.functional.pad(dt[:, :, 1:], (0, 1), value=0.0)
    trap_shifted = torch.nn.functional.pad(trap[:, :, 1:], (0, 1), value=0.0)
    shifted_gamma = dt_shifted * (1 - trap_shifted)
    factor = gamma + shifted_gamma
    k = torch.einsum("bsrhn,bhs->bsrhn", k, factor)
    qk_dot = torch.einsum("bsrRh,bhs->bsrRh", qk_dot, shifted_gamma)

    v = torch.einsum("bthd,hrd->btrhd", v, mimo_v)

    def segsum_unstable(x: Tensor) -> Tensor:
        x_segsum = x[..., :, None] - x[..., None, :]
        mask = torch.tril(torch.ones(x.size(-1), x.size(-1), device=x.device, dtype=torch.bool), diagonal=0)
        return x_segsum.masked_fill(~mask, -torch.inf)

    mimo_mask_outer = segsum_unstable(dA_cs)
    mimo_mask_inner = torch.ones(mimo_rank, mimo_rank, dtype=torch.bool, device=q.device)

    mimo_mask = mimo_mask_outer[..., :, :, None, None] * mimo_mask_inner
    mimo_mask = mimo_mask.reshape(
        *mimo_mask_outer.shape[:-2],
        mimo_mask_outer.shape[-2] * mimo_mask_inner.shape[0],
        mimo_mask_outer.shape[-1] * mimo_mask_inner.shape[1],
    )
    q = rearrange(q, "b (n c) r h d -> b h n (c r) d", c=chunk_size)
    k_scaled = rearrange(k, "b (n c) r h d -> b h n c r d", c=chunk_size)
    k_scaled = torch.einsum("bhncrd,bhnc->bhncrd", k_scaled, torch.exp(dA_cs_rev))
    k_scaled = rearrange(k_scaled, "b h n c r d -> b h n (c r) d", c=chunk_size)
    k = rearrange(k, "b (n c) r h d -> b h n (c r) d", c=chunk_size)
    v = rearrange(v, "b (n c) r h d -> b h n (c r) d", c=chunk_size)
    kv = k_scaled.transpose(-1, -2) @ v

    curr_state = torch.zeros_like(kv[:, :, 0, :, :])
    for n in range(nchunks):
        curr_dA_sum = dA_cs[:, :, n, -1]
        next_state = (torch.exp(curr_dA_sum[:, :, None, None]) * curr_state) + kv[:, :, n, :, :]
        kv[:, :, n, :, :] = curr_state
        curr_state = next_state

    if return_final_state:
        final_state = next_state.float()
    else:
        final_state = None

    q_inter = q * torch.exp(repeat(dA_cs, "b h n c -> b h n (c r)", r=mimo_rank).unsqueeze(-1))
    inter = q_inter @ kv
    intra = ((q @ k.transpose(-1, -2)) * torch.exp(mimo_mask)) @ v
    o = inter + intra
    o = rearrange(o, "b h n (c r) d -> b h n c r d", r=mimo_rank)

    v = rearrange(v, "b h n (c r) d -> b h (n c) r d", r=mimo_rank)
    qk_dot = rearrange(qk_dot, "b t R r h -> b h t R r")
    qkv = torch.einsum("bhtRr,bhtrp->bhtRp", qk_dot, v)
    qkv = rearrange(qkv, "b h (n c) r d -> b h n c r d", c=chunk_size)
    o -= qkv

    if D is not None:
        vd = torch.einsum("bhtrp,h->bhtrp", v, D)
        vd = rearrange(vd, "b h (n c) r d -> b h n c r d", c=chunk_size)
        o += vd

    if z is not None:
        z = torch.einsum("bthd,hrd->btrhd", z, mimo_z)
        z = rearrange(z, "b (n c) r h d -> b h n c r d", c=chunk_size)
        o = o * torch.nn.functional.silu(z)

    if contract_mimo_out:
        if mimo_o is None:
            raise ValueError("mimo_o cannot be None when contract_mimo_out is True")
        o = torch.einsum("bhncrd,hrd->bhncd", o, mimo_o)
        return rearrange(o, "b h n c d -> b (n c) h d"), final_state, final_k

    return rearrange(o, "b h n c r d -> b (n c) r h d"), final_state, final_k


def apply_angle_dt_reference(
    angle: Tensor, # (batch, seqlen, nheads, dim)
    dt: Tensor, # (batch, seqlen, nheads)
    ) -> Tensor:
    base_vals = angle.to(torch.float32)
    base_vals = torch.tanh(base_vals) * dt[..., None].to(torch.float32) * torch.pi
    return torch.cumsum(base_vals, dim=1)


def dacs_segsum_torch(da, chunk_size):
    """
    da: [B, H, S]
    return:
    da_cs: [B, H, S]
    da_cs_rev: [B, H, S]
    segsum: [B, H, num_chunks, C, C]
    """
    B, H, S = da.shape
    C = chunk_size
    num_chunks = (S + C - 1) // C

    pad_len = num_chunks * C - S
    if pad_len > 0:
        da = torch.nn.functional.pad(da, (0, pad_len))

    da = da.view(B, H, num_chunks, C)

    # -----------------------
    # 1. da_cs (forward cumsum)
    # -----------------------
    da_cs = torch.cumsum(da, dim=-1)
    da_cs = torch.minimum(da_cs, torch.zeros_like(da_cs))

    # -----------------------
    # 2. da_cs_rev
    # -----------------------
    da_cs_rev_full = torch.cumsum(da.flip(-1), dim=-1).flip(-1)

    # "roll one to the left"
    da_cs_rev = torch.zeros_like(da_cs_rev_full)
    da_cs_rev[..., :-1] = da_cs_rev_full[..., 1:]
    da_cs_rev = torch.minimum(da_cs_rev, torch.zeros_like(da_cs_rev))

    # -----------------------
    # 3. segsum
    # -----------------------
    i = torch.arange(C, device=da.device).view(C, 1)
    j = torch.arange(C, device=da.device).view(1, C)
    mask = (i > j)  # [C, C]

    seg = da.unsqueeze(-1).expand(-1, -1, -1, -1, C)

    seg = torch.where(mask, seg, torch.zeros_like(seg))

    segsum = torch.cumsum(seg, dim=-2)
    segsum = torch.minimum(segsum, torch.zeros_like(segsum))

    da_cs = da_cs.view(B, H, -1)[..., :S]
    da_cs_rev = da_cs_rev.view(B, H, -1)[..., :S]

    return da_cs, da_cs_rev, segsum