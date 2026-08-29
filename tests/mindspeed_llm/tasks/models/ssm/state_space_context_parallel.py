import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def allgather_async(tensor: torch.Tensor,
                    group: dist.ProcessGroup,
                    size: int):
    """
    Gather `tensor` from every rank in `group` asynchronously, returns
    work handler and buffer with `tensor` from each rank 
    """
    if size == 1:
        return None, tensor

    tensor = tensor.contiguous()
    gather_buf = torch.empty(
        (size, *tensor.shape), dtype=tensor.dtype, device=tensor.device
    )

    work = dist.all_gather_into_tensor(
        gather_buf, tensor, group=group, async_op=True
    )
    return work, gather_buf


class SequenceParallelConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                xBC_curr_rank,
                dt_curr_rank,
                conv1d_weight,
                conv1d_bias,
                dt_bias,
                cp_group,
                cp_size,
                cp_rank,
                kernel_sz,
                nheads,
                d_inner,
                d_state,
                ngroups):
        """
        Implements the forward pass for a sequence-parallel 1D convolution.

        This function handles the communication required for causal convolutions
        across different ranks in a context-parallel group. It overlaps the
        asynchronous communication (`all_gather` of convolution "tails") with
        independent computations (processing of the dt tensor) to improve
        performance.

        Args:
            ctx: The context object for `torch.autograd.Function` to save tensors for backward.
            xBC_curr_rank (Tensor): The local shard of the main input tensor for the current rank.
                                    Shape: [B, L_local, D_xBC], where B is batch size, L_local is the
                                    local sequence length, and D_xBC is the feature dimension for
                                    the combined x, B, and C tensors of the SSM.
            dt_curr_rank (Tensor):  The local shard of the timestep tensor (delta t, or Δt).
                                    Shape: [B, L_local, nheads].
            conv1d_weight (Tensor): The weight parameter of the nn.Conv1d layer.
                                    Shape: [channels, 1, kernel_size].
            conv1d_bias (Tensor):   The bias parameter of the nn.Conv1d layer. Can be None.
                                    Shape: [channels].
            dt_bias (Tensor):       A learnable bias parameter added to the dt tensor.
                                    Shape: [nheads].
            cp_group (ProcessGroup): The process group for context-parallel communication.
            cp_size (int):          The world size of the context-parallel group.
            cp_rank (int):          The rank of the current process in the context-parallel group.
            kernel_sz (int):        The kernel size of the 1D convolution (d_conv).
            nheads (int):           Number of attention heads.
            d_inner (int):          The inner dimension of the Mamba block.
            d_state (int):          The state dimension (N) of the SSM.
            ngroups (int):          Number of groups for B and C parameters.

        Returns:
            xBC_processed (Tensor): The output tensor for the current rank after the sequence-parallel
                                    convolution and SiLU activation. Shape: [B, L_local, D_xBC].
            dt_processed (Tensor):  The processed dt tensor after adding the bias and applying softplus.
                                    Shape: [B, L_local, nheads].
        """


        tail_len = kernel_sz - 1

        # 1. Prepare local_tail for all_gather
        local_tail_for_ag = None # Initialize
        if cp_size > 1 and tail_len > 0:
            local_tail_for_ag = xBC_curr_rank[:, -tail_len:, :].contiguous()
        elif cp_size > 1 and tail_len == 0: # Should not happen with kernel_sz > 1
            # Create an empty tensor with correct batch and feature dims, but 0 seq_len
            local_tail_for_ag = torch.empty((xBC_curr_rank.shape[0], 0, xBC_curr_rank.shape[2]),
                                            dtype=xBC_curr_rank.dtype, device=xBC_curr_rank.device)

        # 2. Initiate Asynchronous AllGather for local_tail_for_ag
        ag_work_handle, ag_buf = None, None
        if tail_len > 0:  # Only do all_gather if there's actually a tail
            ag_work_handle, ag_buf = allgather_async(local_tail_for_ag, cp_group, cp_size)
        elif cp_size > 1:  # tail_len is 0 but cp_size > 1
            ag_work_handle, ag_buf = allgather_async(local_tail_for_ag, cp_group, cp_size)

        # 3. Perform Computation A (independent of all_gather result - dt processing)
        dt_contiguous = dt_curr_rank.contiguous()
        dt_plus_bias = dt_contiguous + dt_bias
        dt_processed = F.softplus(dt_plus_bias)

        # 4. Wait for all_gather to complete (if it was launched)
        if ag_work_handle:
            ag_work_handle.wait()

        # 5. Prepare input for convolution using gathered tails
        prev_tail_data = None
        if tail_len > 0:
            if cp_size == 1:
                prev_tail_data = torch.zeros_like(local_tail_for_ag) # Padded with zeros
            elif cp_rank == 0:
                prev_tail_data = torch.zeros_like(local_tail_for_ag) # Padded with zeros
            else:
                prev_tail_data = ag_buf[cp_rank - 1]
            conv_input = torch.cat([prev_tail_data, xBC_curr_rank], dim=1)
        else:
            conv_input = xBC_curr_rank

        # 6. Perform Convolution
        conv_input_transposed = conv_input.transpose(1, 2).contiguous()
        padding_val = 0
        conv_output_transposed = F.conv1d(
            conv_input_transposed,
            conv1d_weight,
            conv1d_bias,
            stride=1,
            padding=padding_val,
            dilation=1,
            groups=conv1d_weight.shape[0]
        )
        conv_output_full = conv_output_transposed.transpose(1, 2)
        xBC_conv_sliced = conv_output_full
        xBC_processed = F.silu(xBC_conv_sliced.contiguous())

        # Save tensors for backward
        ctx.save_for_backward(
            xBC_curr_rank, dt_contiguous, local_tail_for_ag, ag_buf, prev_tail_data, conv_input,
            conv1d_weight, conv1d_bias, dt_bias, dt_plus_bias, xBC_conv_sliced
        )
        # Save non-tensor attributes
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank
        ctx.kernel_sz = kernel_sz
        ctx.tail_len = tail_len
        ctx.padding_val = padding_val
        ctx.conv_groups = conv1d_weight.shape[0]
        ctx.xBC_curr_rank_seqlen = xBC_curr_rank.shape[1]
        ctx.xBC_curr_rank_requires_grad = xBC_curr_rank.requires_grad
        ctx.conv1d_weight_requires_grad = conv1d_weight.requires_grad
        ctx.conv1d_bias_requires_grad = conv1d_bias is not None and conv1d_bias.requires_grad

        ctx.dt_curr_rank_requires_grad = dt_curr_rank.requires_grad
        ctx.dt_bias_requires_grad = dt_bias.requires_grad

        return xBC_processed, dt_processed

    @staticmethod
    def backward(ctx, grad_xBC_processed, grad_dt_processed):
        (
            xBC_curr_rank, dt_contiguous_saved, local_tail_saved, ag_buf_saved, prev_tail_saved, conv_input_saved,
            conv1d_weight_saved, conv1d_bias_saved, dt_bias_saved, dt_plus_bias_saved, xBC_conv_sliced_saved
        ) = ctx.saved_tensors

        # Backward for Computation B (Convolution and SiLU path)

        # 1. SiLU backward via official operator for perfect alignment
        x_slice = xBC_conv_sliced_saved.contiguous()
        grad_silu = torch.ops.aten.silu_backward(
            grad_xBC_processed.contiguous(),
            x_slice
        )[0]

        # 2. Reconstruct full conv1d output gradient
        B, L_in, C = conv_input_saved.shape
        L_out = L_in - ctx.tail_len
        grad_conv_out = conv_input_saved.new_zeros((B, L_out, C))
        grad_conv_out[:, :, :] = grad_silu

        # 3. Convert to (B, C, L_out) for conv1d grad
        grad_out_tc = grad_conv_out.transpose(1, 2).contiguous()  # (B, C_out, L_out)

        # 4. Compute grad w.r.t. conv input & weight via high-level API
        inp_tc = conv_input_saved.transpose(1, 2).contiguous()    # (B, C_in, L_in)
        grad_conv_input_tc = torch.nn.grad.conv1d_input(
            input_size=inp_tc.shape,
            weight=conv1d_weight_saved,
            grad_output=grad_out_tc,
            stride=1,
            padding=ctx.padding_val,
            dilation=1,
            groups=ctx.conv_groups,
        )
        grad_conv1d_weight_val = torch.nn.grad.conv1d_weight(
            input=inp_tc,
            weight_size=conv1d_weight_saved.shape,
            grad_output=grad_out_tc,
            stride=1,
            padding=ctx.padding_val,
            dilation=1,
            groups=ctx.conv_groups,
        )
        grad_conv1d_bias_val = None
        if conv1d_bias_saved is not None and ctx.conv1d_bias_requires_grad:
            grad_conv1d_bias_val = grad_out_tc.sum(dim=(0, 2))  # sum over batch & length

        # 5. Convert grad_conv_input back to (B, L_in, C)
        grad_conv_input = grad_conv_input_tc.transpose(1, 2)

        # 6. Split into prefix tail and main body
        if ctx.tail_len > 0:
            grad_prev_tail = grad_conv_input[:, :ctx.tail_len, :]
            grad_xBC_from_conv = grad_conv_input[:, ctx.tail_len:, :]
        else:
            grad_prev_tail = torch.empty((B, 0, C), device=grad_conv_input.device, dtype=grad_conv_input.dtype)
            grad_xBC_from_conv = grad_conv_input

        # Gradients for AllGather input (local_tail_saved) via ReduceScatter
        grad_local_tail_scattered = None
        rs_handle = None
        if ctx.cp_size > 1 and ctx.tail_len > 0 and ctx.xBC_curr_rank_requires_grad:
            # prepare grad_buf of shape (cp_size, B, tail_len, C)
            grad_buf = torch.zeros_like(ag_buf_saved, dtype=grad_prev_tail.dtype)
            if ctx.cp_rank > 0:
                grad_buf[ctx.cp_rank - 1] = grad_prev_tail

            grad_local_tail_scattered = torch.empty_like(local_tail_saved)
            if grad_local_tail_scattered.numel() > 0:
                rs_handle = dist.reduce_scatter_tensor(
                    output=grad_local_tail_scattered,
                    input=grad_buf,
                    op=dist.ReduceOp.SUM,
                    group=ctx.cp_group,
                    async_op=True
                )

        # Gradients for Computation A (dt_processed path)
        grad_dt_plus_bias = grad_dt_processed * torch.sigmoid(dt_plus_bias_saved)
        grad_dt_curr = grad_dt_plus_bias
        grad_dt_bias = None
        if ctx.dt_bias_requires_grad:
            dims = list(range(grad_dt_plus_bias.ndim - dt_bias_saved.ndim))
            summed = grad_dt_plus_bias.sum(dim=dims)
            grad_dt_bias = summed.reshape(dt_bias_saved.shape)

        # 7. Wait for scatter to finish
        if rs_handle is not None:
            rs_handle.wait()

        # 8. Combine xBC gradients
        grad_xBC_total = None
        if ctx.xBC_curr_rank_requires_grad:
            grad_xBC_total = torch.zeros_like(xBC_curr_rank)
            grad_xBC_total += grad_xBC_from_conv
            if grad_local_tail_scattered is not None and grad_local_tail_scattered.numel() > 0:
                actual = min(ctx.tail_len, xBC_curr_rank.shape[1])
                if actual > 0:
                    part = grad_local_tail_scattered[:, -actual:, :]
                    grad_xBC_total[:, -actual:, :] += part

        # 9. Zero out grads for non-requires
        final_dt = grad_dt_curr
        final_w = grad_conv1d_weight_val
        final_b = grad_conv1d_bias_val
        final_db = grad_dt_bias

        return (
            grad_xBC_total,
            final_dt,
            final_w,
            final_b,
            final_db,
            None, None, None, None, None, None, None, None
        )


class SSDAllgatherOverlapFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                local_decay,
                local_hidden_state,
                C_ssd_chunked_b,
                B_ssd_chunked_b,
                x_ssd_chunked,
                A_ssd_reshaped,
                A_cumsum_ssd,
                cp_group,
                cp_size,
                segsum_fn_ref,
                device_for_ops):
        """
        Implements the forward pass for the core sequence-parallel SSM computation.

        This function orchestrates the communication-computation overlap central to the
        sequence-parallel Structured State Space (SSD) algorithm. It initiates an
        asynchronous all-gather of the per-rank state contributions (local_decay and
        local_hidden_state) across the context-parallel group. While this
        communication is in flight, it computes the independent, intra-chunk (diagonal)
        part of the SSM output, Y_diag.

        Args:
            ctx: The context object for `torch.autograd.Function` to save tensors for backward.
            local_decay (Tensor): The per-rank decay factor (Λ_r). It represents how much the
                                  hidden state decays over the entire sequence segment processed
                                  by the current rank. Shape: [B, H].
            local_hidden_state (Tensor): The per-rank hidden state contribution (H_r). It is the state
                                         accumulated over the current rank's sequence, assuming a
                                         zero input state. Shape: [B, H, P, N].
            C_ssd_chunked_b (Tensor): The pre-transformed (permuted and reshaped) C parameter tensor
                                      for the current rank's chunks, optimized for BMM.
                                      Shape: [B*H*C_chunks, L_chunk, N].
            B_ssd_chunked_b (Tensor): The pre-transformed (permuted, reshaped, and transposed) B
                                      parameter tensor for the current rank's chunks.
                                      Shape: [B*H*C_chunks, N, L_chunk].
            x_ssd_chunked (Tensor):   The chunked and discretized input tensor x for the current rank.
                                      Shape: [B, C_chunks, L_chunk, H, P].
            A_ssd_reshaped (Tensor):  The chunked and reshaped Δt * A tensor, used to compute the
                                      L_ij matrix for the Y_diag calculation.
                                      Shape: [B, H, C_chunks, L_chunk].
            A_cumsum_ssd (Tensor):    The cumulative sum of A_ssd_reshaped. Used to compute the
                                      intra-chunk decay for the Y_off calculation.
                                      Shape: [B, H, C_chunks, L_chunk].
            cp_group (ProcessGroup):  The process group for context-parallel communication.
            cp_size (int):            The world size of the context-parallel group.
            segsum_fn_ref (function): A reference to the function for computing the segmented sum
                                      (used to create the L_ij matrix).
            device_for_ops (torch.device): The target device for creating new tensors if needed.

        Returns:
            ld_buf (Tensor):           The buffer containing `local_decay` gathered from all ranks.
                                       Shape: [cp_size, B, H].
            lhs_buf (Tensor):          The buffer containing `local_hidden_state` gathered from all ranks.
                                       Shape: [cp_size, B, H, P, N].
            Y_diag_computed (Tensor):  The diagonal component (Y_diag) of the SSM output for the
                                       current rank. Shape: [B, C_chunks, L_chunk, H, P].
            state_decay_out_computed (Tensor): The intra-chunk state decay factor (exp(A_cumsum)),
                                               used for the Y_off calculation.
                                               Shape: [B, H, C_chunks, L_chunk].
        """

        # 1. Initiate Asynchronous AllGathers
        work_ld, ld_buf = allgather_async(local_decay, cp_group, cp_size)
        work_lhs, lhs_buf = allgather_async(local_hidden_state, cp_group, cp_size)

        # 2. Perform "Independent" Forward Computations
        s_val_for_L = segsum_fn_ref(A_ssd_reshaped)
        L_val = torch.exp(s_val_for_L)

        # Start of Y_diag_computed calculation using bmm approach
        B_dim = x_ssd_chunked.shape[0]
        C_chunks_dim = x_ssd_chunked.shape[1]
        L_chunk_dim = x_ssd_chunked.shape[2]
        H_heads_dim = x_ssd_chunked.shape[3]
        N_state_dim = B_ssd_chunked_b.shape[1]
        P_headdim_dim = x_ssd_chunked.shape[4]

        # 1. Permute x_ssd_chunked to align with ssdOrigin's C_r, B_r, x_r
        x_r_like = x_ssd_chunked.permute(0, 3, 1, 2, 4).contiguous()  # (B, H_heads, C_chunks, L_chunk, P_headdim)

        # 2. Reshape for batch matrix multiplication (bmm)
        x_b_like = x_r_like.reshape(-1, L_chunk_dim, P_headdim_dim)
        L_b_like = L_val.to(torch.bfloat16).reshape(-1, L_val.shape[3], L_val.shape[4])

        # 3. Perform bmm operations as in ssdOrigin
        CB_b_like = torch.bmm(C_ssd_chunked_b, B_ssd_chunked_b)
        CBL_b_like = (CB_b_like * L_b_like).to(torch.bfloat16)
        Y_diag_intermediate = torch.bmm(CBL_b_like, x_b_like)

        # 4. Reshape and permute Y_diag_intermediate to the final target shape
        Y_diag_computed_temp = Y_diag_intermediate.reshape(
            B_dim, H_heads_dim, C_chunks_dim, L_chunk_dim, P_headdim_dim
        )
        # Permute to target shape (B, C_chunks, L_chunk, H_heads, P_headdim)
        Y_diag_computed = Y_diag_computed_temp.permute(0, 2, 3, 1, 4).contiguous()

        # End of Y_diag_computed calculation using bmm approach

        state_decay_out_computed = torch.exp(A_cumsum_ssd).to(torch.bfloat16).contiguous()

        # 3. Wait for AllGathers
        if work_ld:
            work_ld.wait()
        if work_lhs:
            work_lhs.wait()

        # Save tensors for backward pass
        ctx.save_for_backward(local_decay, local_hidden_state, ld_buf, lhs_buf,
                              C_ssd_chunked_b, B_ssd_chunked_b, x_ssd_chunked,
                              A_ssd_reshaped, A_cumsum_ssd,
                              L_val,
                              s_val_for_L,
                              state_decay_out_computed
                              )
        # Save attributes
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size

        # For manual segsum backward
        ctx.T_for_segsum = A_ssd_reshaped.size(-1)
        ctx.device_for_ops = device_for_ops

        return ld_buf, lhs_buf, Y_diag_computed, state_decay_out_computed


    @staticmethod
    def backward(ctx,
                 grad_ld_buf, grad_lhs_buf,
                 grad_Y_diag_computed, grad_state_decay_out_computed):

        (local_decay_saved, local_hidden_state_saved, ld_buf_saved, lhs_buf_saved,
         C_b_s, B_b_s, x_ssd_chunked_saved,
         A_ssd_reshaped_saved, A_cumsum_ssd_saved,
         L_val_saved, s_val_for_L_saved, state_decay_out_saved
        ) = ctx.saved_tensors

        # 1. Initiate Asynchronous ReduceScatters
        grad_local_decay_rs_output = None
        rs_handle_ld = None
        if ctx.cp_size > 1:
            if local_decay_saved.numel() > 0:
                grad_local_decay_rs_output = torch.empty_like(local_decay_saved)
                rs_handle_ld = dist.reduce_scatter_tensor(grad_local_decay_rs_output, grad_ld_buf, group=ctx.cp_group, async_op=True)
            else:
                grad_local_decay_rs_output = torch.empty_like(local_decay_saved)
        else:
            grad_local_decay_rs_output = grad_ld_buf.clone() if grad_ld_buf is not None else None

        grad_local_hidden_state_rs_output = None
        rs_handle_lhs = None
        if ctx.cp_size > 1:
            if local_hidden_state_saved.numel() > 0:
                grad_local_hidden_state_rs_output = torch.empty_like(local_hidden_state_saved)
                rs_handle_lhs = dist.reduce_scatter_tensor(grad_local_hidden_state_rs_output, grad_lhs_buf, group=ctx.cp_group, async_op=True)
            else:
                grad_local_hidden_state_rs_output = torch.empty_like(local_hidden_state_saved)
        else:
            grad_local_hidden_state_rs_output = grad_lhs_buf.clone() if grad_lhs_buf is not None else None

        # 2. DIRECT Gradient Computations (Overlapped)
        grad_C_ssd_chunked_b = None
        grad_B_ssd_chunked_b = None
        grad_x_ssd_chunked = None
        grad_A_cumsum_ssd = None
        grad_L_val_intermediate = None       # d(Loss)/d(L_val)

        # Gradients for Y_diag_computed inputs (using bmm decomposition)
        if grad_Y_diag_computed is not None:
            gY_out = grad_Y_diag_computed    # Shape: (B, C_chunks, L_chunk, H_heads, P_headdim)

            X_in_s = x_ssd_chunked_saved
            L_in_s = L_val_saved

            # Get dimensions from saved tensors
            B_dim = X_in_s.shape[0]
            C_chunks_dim = X_in_s.shape[1]
            L_chunk_dim = X_in_s.shape[2]
            H_heads_dim = X_in_s.shape[3]
            P_headdim_dim = X_in_s.shape[4]
            B_eff = C_b_s.shape[0]           # B * H * C_chunks

            # Inverse permutation (0,3,1,2,4) maps (B,C,L,H,P) back to (B,H,C,L,P)
            gY_temp_reshaped = gY_out.permute(0, 3, 1, 2, 4)

            # Backward through Y_temp_reshaped = Y_inter_b.reshape(...)
            gY_inter_b = gY_temp_reshaped.reshape(B_eff, L_chunk_dim, P_headdim_dim)

            # Recompute intermediates from forward pass needed for bmm backward
            X_r_like_s = X_in_s.permute(0, 3, 1, 2, 4)
            X_b_s = X_r_like_s.reshape(B_eff, L_chunk_dim, P_headdim_dim)

            L_b_s = L_in_s.to(torch.bfloat16).reshape(B_eff, L_chunk_dim, L_chunk_dim)

            CB_b_s = torch.bmm(C_b_s, B_b_s)                  # (B_eff, L_chunk, L_chunk)
            CBL_b_s = (CB_b_s * L_b_s).to(torch.bfloat16)     # (B_eff, L_chunk, L_chunk)

            # Backward through Y_inter_b = torch.bmm(CBL_b_s, X_b_s)
            gCBL_b = torch.bmm(gY_inter_b, X_b_s.transpose(1, 2))
            gX_b = torch.bmm(CBL_b_s.transpose(1, 2), gY_inter_b).contiguous()
            gX_r_like = gX_b.reshape(B_dim, H_heads_dim, C_chunks_dim, L_chunk_dim, P_headdim_dim)
            grad_x_ssd_chunked = gX_r_like.permute(0, 2, 3, 1, 4).contiguous()

            # Backward through CBL_b_s = CB_b_s * L_b_s (element-wise)
            gCB_b_from_CBL = None
            if gCBL_b is not None:
                gCB_b_from_CBL = (gCBL_b * L_b_s).to(torch.bfloat16)
                gL_b = (gCBL_b * CB_b_s).to(torch.bfloat16).contiguous()
                grad_L_val_intermediate = gL_b.reshape(B_dim, H_heads_dim, C_chunks_dim, L_chunk_dim, L_chunk_dim)

            # Backward through CB_b_s = torch.bmm(C_b_s, B_b_s)
            if gCB_b_from_CBL is not None:
                grad_C_ssd_chunked_b = torch.bmm(gCB_b_from_CBL, B_b_s.transpose(1, 2))
                grad_B_ssd_chunked_b = torch.bmm(C_b_s.transpose(1, 2), gCB_b_from_CBL)

        # Gradient for A_ssd_reshaped (from L_val's gradient)
        grad_A_ssd_reshaped = None
        if grad_L_val_intermediate is not None:
            grad_s_val = (grad_L_val_intermediate * L_val_saved)
            T_mask_dim = A_ssd_reshaped_saved.size(-1)
            if s_val_for_L_saved.shape[-1] != T_mask_dim or s_val_for_L_saved.shape[-2] != T_mask_dim:
                raise ValueError(f"T_mask_dim ({T_mask_dim}) mismatch with s_val_for_L_saved dimensions "
                                 f"({s_val_for_L_saved.shape[-2]}, {s_val_for_L_saved.shape[-1]})")
            mask_triu = torch.tril(torch.ones(T_mask_dim, T_mask_dim, dtype=torch.bool, device=ctx.device_for_ops), diagonal=0)
            mask_triu_bc = mask_triu.view((1,) * (grad_s_val.ndim - 2) + (T_mask_dim, T_mask_dim))
            grad_x_sc_inter = grad_s_val.masked_fill(~mask_triu_bc, 0.0)
            grad_x_masked_tril_dim_minus_2 = torch.cumsum(torch.flip(grad_x_sc_inter, dims=[-2]), dim=-2)
            grad_x_masked_tril = torch.flip(grad_x_masked_tril_dim_minus_2, dims=[-2])
            mask_tril = torch.tril(torch.ones(T_mask_dim, T_mask_dim, dtype=torch.bool, device=ctx.device_for_ops), diagonal=-1)
            mask_tril_bc = mask_tril.view((1,) * (grad_x_masked_tril.ndim - 2) + (T_mask_dim, T_mask_dim))
            grad_x_rep = grad_x_masked_tril.masked_fill(~mask_tril_bc, 0.0)
            grad_A_ssd_reshaped = grad_x_rep.sum(dim=-1)

        # Gradient for A_cumsum_ssd
        if grad_state_decay_out_computed is not None:
            grad_A_cumsum_ssd = (grad_state_decay_out_computed * state_decay_out_saved).to(torch.bfloat16)

        # 3. Wait for ReduceScatters
        if rs_handle_ld:
            rs_handle_ld.wait()
        if rs_handle_lhs:
            rs_handle_lhs.wait()

        return (
            grad_local_decay_rs_output,
            grad_local_hidden_state_rs_output,
            grad_C_ssd_chunked_b,
            grad_B_ssd_chunked_b,
            grad_x_ssd_chunked,
            grad_A_ssd_reshaped,
            grad_A_cumsum_ssd,
            None, None, None, None
        )
