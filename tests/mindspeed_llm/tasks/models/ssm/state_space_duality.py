from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
from megatron.core import mpu

from mindspeed_llm.tasks.models.ssm.state_space_context_parallel import SSDAllgatherOverlapFn


@dataclass
class ProcessInputs:
    """Input data structure for main processing flow"""
    x: torch.Tensor        # (B, L, H, P)
    dt: torch.Tensor       # (B, L, H)
    A: torch.Tensor        # (H,)
    B: torch.Tensor        # (B, L, G, S)
    C: torch.Tensor        # (B, L, G, S)
    D: torch.Tensor        # Residual matrix


class StateOptions:
    def __init__(self, initial_states=None, return_final_state=False, cached_start=False):
        """State management options"""
        self.initial_states = initial_states
        self.return_final_state = return_final_state
        self.cached_start = cached_start
        self.final_state = None  # Added for state persistence

    @property
    def should_return_final(self) -> bool:
        """Determine if final state should be returned"""
        return self.return_final_state or self.cached_start


class StateSpaceProcessor:
    def __init__(self, config):
        """
        Configuration should contain:
        - nheads_local: Number of local heads
        - ngroups_local: Number of local groups
        - dt_min/dt_max: Time step constraints
        - dt_bias: Time step bias term
        - headdim: Dimension per head
        - d_state: State dimension
        - chunk_size: Processing chunk size
        - D_has_hdim: Dimension for D matrix
        """
        self.config = self._validate_config(config)

    @property
    def h_ratio(self) -> int:
        return self.config['nheads_local'] // self.config['ngroups_local']

    def _validate_config(self, config) -> dict:
        """Configuration validation"""
        required_keys = {'nheads_local', 'ngroups_local', 'dt_min', 'dt_max', 'dt_bias',
                        'headdim', 'd_state', 'chunk_size', 'D_has_hdim'}
        missing = required_keys - config.keys()
        if missing:
            raise ValueError(f"Missing config keys: {missing}")
        return config

    def process(self, inputs: ProcessInputs, state_opts: StateOptions = StateOptions()):
        """
        Main processing pipeline
        Args:
            inputs: Input data
            state_opts: State options
        Returns:
            y: (B, L, H, P) Output features
        """
        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()
        cp_group = mpu.get_context_parallel_group()

        # Unpack inputs
        x, dt, A, B, C, D = inputs.x, inputs.dt, inputs.A, inputs.B, inputs.C, inputs.D

        # Parameter initialization
        initial_states = self._prepare_initial_states(state_opts.initial_states)
        seq_len = x.size(1)
        pad_size = self._calculate_padding(seq_len)

        # Dimension transformations
        x, dt, A, B, C = self._expand_dims(x, A, dt, B, C)
        B_exp, C_exp = self._expand_groups_to_heads(B, C)
        D = self._prepare_residual(D, x, pad_size)

        if cp_size == 1:
            dt_proc = self._process_time_step(dt)

            # Chunk processing
            x_pad, A_pad, B_pad, C_pad = self._chunk_and_pad(x, dt_proc, A, B_exp, C_exp, pad_size)

            # Core computations
            Y_diag, states, A_cum, C_br = self._compute_diagonal_blocks(A_pad, B_pad, C_pad, x_pad)
            Y_off, final_state = self._compute_inter_chunk_blocks(A_cum, C_br, states, initial_states)

            # Output synthesis
            state_opts.final_state = final_state

        elif cp_size > 1:
            self.config['dt_min'] = torch.tensor(self.config['dt_min'], dtype=torch.float32, device=dt.device)
            self.config['dt_max'] = torch.tensor(self.config['dt_max'], dtype=torch.float32, device=dt.device)
            dt_proc = torch.clamp(dt, self.config['dt_min'], self.config['dt_max'])

            x_c, A_c, B_c, C_c = self._chunk_and_pad(x, dt_proc, A, B_exp, C_exp, pad_size)
            device = x_c.device
            A_reshaped = rearrange(A_c, "b c l h -> b h c l").contiguous()          # (B, H, C, L)

            A_cumsum = torch.cumsum(A_reshaped, dim=-1).contiguous()                # (B, H, C, L)

            decay_states_arg = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum).to(torch.bfloat16)  # (B,H,C,L) used with B_c, x_c
            states_calc = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B_c, decay_states_arg, x_c)

            if initial_states is not None:
                initial_states_loop_val = initial_states
            else:
                initial_states_loop_val = torch.zeros_like(states_calc[:, :1], dtype=torch.float32) # (B, 1, H, P, N)

            padded_A_cumsum_end = F.pad(A_cumsum[:, :, :, -1], (1, 0))
            decay_chunk_calc = torch.exp(self._segmented_sum(padded_A_cumsum_end.contiguous()))  # (B,H,C+1,C+1)

            local_decay = decay_chunk_calc[:, :, -1, 0].contiguous()                # (B, H)

            new_states_calc = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk_calc[:, :, 1:, 1:].to(torch.bfloat16), states_calc).contiguous()
            # Split into two parts: the first C_chunks-1 elements, and the last 1 element.
            C_chunks = new_states_calc.shape[1]
            new_states_prefix, local_hidden_state_unsqeezed = torch.split(
                new_states_calc,
                [C_chunks - 1, 1],
                dim=1
            )
            # We squeeze it to match the original shape (B, H, P, N).
            local_hidden_state = local_hidden_state_unsqeezed.squeeze(1).contiguous()

            L_chunk_dim_ssp = C_c.shape[2]
            N_state_dim_ssp = C_c.shape[4]
            C_r = C_c.permute(0, 3, 1, 2, 4)
            C_b = C_r.reshape(-1, L_chunk_dim_ssp, N_state_dim_ssp)

            B_r = B_c.permute(0, 3, 1, 2, 4)                                         # (B, H_heads, C_chunks, L_chunk, N_state)
            B_b = B_r.reshape(-1, L_chunk_dim_ssp, N_state_dim_ssp).transpose(1, 2)  # (B*H*C_chunks, N_state, L_chunk)

            ld_buf, lhs_buf, Y_diag, state_decay_out = SSDAllgatherOverlapFn.apply(
                local_decay, local_hidden_state,
                C_b, B_b, x_c, A_reshaped, A_cumsum,
                cp_group, cp_size,
                self._segmented_sum, device
            )

            if cp_rank > 0:
                prod_terms_for_i = [None] * cp_rank
                prod_terms_for_i[cp_rank - 1] = torch.ones_like(ld_buf[0])          # ld_buf[0] is (B,H)
                if cp_rank > 1:
                    decays_slice_S = ld_buf[1:cp_rank]
                    if decays_slice_S.numel() > 0:
                        # 1. Compute cumulative product on the flipped tensor, but do NOT flip it back.
                        cumprod_from_end = torch.cumprod(torch.flip(decays_slice_S, dims=[0]), dim=0)
                        # 2. Assign to the list using a reverse-indexed loop.
                        num_elements = cp_rank - 1
                        for i in range(num_elements):
                            prod_terms_for_i[i] = cumprod_from_end[num_elements - 1 - i]

                for i in range(cp_rank):
                    current_decay_prod = prod_terms_for_i[i]
                    term_to_add = torch.einsum("bh, bhpn -> bhpn", current_decay_prod, lhs_buf[i])
                    initial_states_loop_val += term_to_add.unsqueeze(1)             # unsqueeze to (B,1,H,P,N) for broadcast

            added_init_state = torch.einsum("bhc, bihpn -> bchpn",
                                            decay_chunk_calc[:, :, :-1, 0],         # (B,H,C) where C is n_chunks
                                            initial_states_loop_val                 # (B,1,H,P,N) broadcast i
                                            ).to(torch.bfloat16).contiguous()

            zeros_for_cat = torch.zeros_like(new_states_prefix[:, :1])                       # (B,1,H,P,N)
            concatenated_new_states = torch.cat([zeros_for_cat, new_states_prefix], dim=1)  # (B, C_chunks, H,P,N)
            off_states = added_init_state + concatenated_new_states.contiguous()

            states_b = off_states.permute(0, 2, 1, 3, 4).reshape(-1, off_states.shape[3], off_states.shape[4]).transpose(-1, -2)
            Cs_b = torch.bmm(C_b, states_b).reshape(C_r.shape[0], C_r.shape[1], C_r.shape[2], C_r.shape[3], states_b.shape[2]).contiguous()
            state_decay_out_us = state_decay_out.unsqueeze(-1).contiguous()
            Y_off = (Cs_b * state_decay_out_us).permute(0, 2, 3, 1, 4).contiguous()

            state_opts.final_state = None                                           # Not handling inference state during training

        return self._synthesize_output((Y_diag, Y_off, D), (pad_size, seq_len), state_opts)

    def _expand_dims(self, x, A, dt, B, C):
        x = rearrange(x, "b l (h p) -> b l h p", p=self.config['headdim']).contiguous()
        dt = dt.contiguous()
        A = A.contiguous()
        B = rearrange(B, "b l (g n) -> b l g n", n=self.config['d_state']).contiguous()
        C = rearrange(C, "b l (g n) -> b l g n", n=self.config['d_state']).contiguous()
        return x, dt, A, B, C

    def _prepare_initial_states(self, states: Optional[torch.Tensor]) -> torch.Tensor:
        """State initialization"""
        return rearrange(states, "b n h p -> b 1 n h p") if states is not None else None

    def _calculate_padding(self, seq_len: int) -> int:
        """Calculate padding length"""
        return (self.config['chunk_size'] - (seq_len % self.config['chunk_size'])) % self.config['chunk_size']

    def _expand_groups_to_heads(self, B, C):
        """Dimension expansion: groups -> heads"""
        B_exp = repeat(B, "b l g d -> b l (g h) d", h=self.h_ratio)
        C_exp = repeat(C, "b l g d -> b l (g h) d", h=self.h_ratio)
        return B_exp, C_exp

    def _process_time_step(self, dt):
        """Time step parameter processing"""
        self.config['dt_min'] = torch.tensor(self.config['dt_min'], dtype=torch.float32, device=dt.device)
        self.config['dt_max'] = torch.tensor(self.config['dt_max'], dtype=torch.float32, device=dt.device)
        dt_proc = nn.functional.softplus(dt + self.config['dt_bias'])
        return torch.clamp(dt_proc, self.config['dt_min'], self.config['dt_max'])

    def _prepare_residual(self, D, x, pad_size):
        """Residual connection preparation"""
        D = rearrange(D.float(), "(h p) -> h p", p=self.config['headdim']) \
                    if self.config['D_has_hdim'] else D
        x_pad = self._pad_sequence(x, pad_size) if pad_size else x
        return rearrange(D, "h -> 1 1 h 1") * x_pad

    def _chunk_and_pad(self, x, dt, A, B, C, pad_size):
        """Chunking and padding operations"""
        # Discretization
        x = x * dt.unsqueeze(-1)
        A = A * dt

        # Padding and chunking
        x, A, B, C = [
            rearrange(
                tensor=self._pad_sequence(tensor, pad_size) if pad_size else tensor,
                pattern="b (c l) ... -> b c l ...",
                l=self.config['chunk_size'] 
            )
            for tensor in (x, A, B, C)
        ]
        return x, A, B, C
        
    def _compute_diagonal_blocks(self, A, B, C, x):
        """Diagonal block computation"""
        A = rearrange(A, "b c l h -> b h c l")
        A_cum = torch.cumsum(A, dim=-1)
        L = torch.exp(self._segmented_sum(A)).to(torch.bfloat16)

        C_r = C.permute(0, 3, 1, 2, 4)
        B_r = B.permute(0, 3, 1, 2, 4)
        x_r = x.permute(0, 3, 1, 2, 4)        
        C_b = C_r.reshape(-1, C_r.shape[3], C_r.shape[4])
        B_b = B_r.reshape(-1, B_r.shape[3], B_r.shape[4]).transpose(1, 2)
        x_b = x_r.reshape(-1, x_r.shape[3], x_r.shape[4])
        CB_b = torch.bmm(C_b, B_b)
        L_b = L.reshape(-1, L.shape[3], L.shape[4])
        CBL_b = CB_b * L_b
        # 对角项计算
        Y_diag = torch.bmm(CBL_b, x_b).reshape(x_r.shape).permute(0, 2, 3, 1, 4).contiguous()

        # 状态初始化
        decay = torch.exp(A_cum[:, :, :, -1:] - A_cum).to(torch.bfloat16)
        decay_states_us = decay.unsqueeze(-1)
        Bd_r = B_r * decay_states_us
        Bd_b = Bd_r.reshape(-1, Bd_r.shape[3], Bd_r.shape[4]).transpose(1, 2)

        states = torch.bmm(Bd_b, x_b).transpose(1, 2).reshape(Bd_r.shape[0], Bd_r.shape[1], Bd_r.shape[2], x_r.shape[4], Bd_r.shape[4]).permute(0, 2, 1, 3, 4).contiguous()
        return Y_diag, states, A_cum, (C_b, C_r)

    def _compute_inter_chunk_blocks(self, A_cum, C_br, states, initial_states):
        """Inter-chunk computation"""
        C_b, C_r = C_br
        # 状态递推
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        decay = torch.exp(self._segmented_sum(nn.functional.pad(A_cum[:, :, :, -1], (1, 0)))).to(torch.bfloat16)
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay, states)
        states, final_state = new_states[:, :-1], new_states[:, -1]

        # 状态转换输出
        state_decay_out = torch.exp(A_cum).to(torch.bfloat16)
        states_b = states.permute(0, 2, 1, 3, 4).reshape(-1, states.shape[3], states.shape[4]).transpose(-1, -2) 
        Cs_b = torch.bmm(C_b, states_b).reshape(C_r.shape[0], C_r.shape[1], C_r.shape[2], C_r.shape[3], states_b.shape[2])
        state_decay_out_us = state_decay_out.unsqueeze(-1)  
        Y_off = (Cs_b * state_decay_out_us).permute(0, 2, 3, 1, 4).contiguous()
        return Y_off, final_state

    def _synthesize_output(
        self,
        y_parts: tuple,          # (Y_diag, Y_off, D)
        seq_meta: tuple,         # (pad_size, seq_len)
        state_opts: StateOptions # Contains final_state and return controls
    ):
        """Unpack tuples"""
        Y_diag, Y_off, D = y_parts
        pad_size, seq_len = seq_meta

        """Get parameters from state options"""
        final_state = state_opts.final_state
        return_final = state_opts.should_return_final

        """Output synthesis"""
        y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
        y = y + D

        if pad_size > 0:
            y = y[:, :seq_len, :, :]

        if not return_final:
            return y
        else:
            return y, final_state

    def _pad_sequence(self, x, pad_size=0):
        """Padding handling"""
        if not 2 < len(x.shape) < 5:
            raise AssertionError('len(x.shape) must in range(2, 5)')
        
        pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(x.shape) == 4 else (0, 0, 0, pad_size, 0, 0)

        return nn.functional.pad(x, pad_shape, mode="constant", value=0)

    def _segmented_sum(self, x):
        """Numerically stable segmented summation"""
        T = x.size(-1)
        x = repeat(x, "... d -> ... d e", e=T)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
        x = x.masked_fill(~mask, 0)
        x_segsum = torch.cumsum(x, dim=-2)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum