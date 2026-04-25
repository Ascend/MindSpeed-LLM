import logging

from mindspeed_llm.fsdp2.distributed.parallel_state import ParallelState

cal_split_sizes = None
all_to_all = None

from mindspeed_llm.fsdp2.distributed.context_parallel.ring_context_parallel.ring_context_parallel import ringattn_context_parallel, \
    ringattn_context_parallel_tnd_general

logger = logging.getLogger(__name__)
_flash_attention_forward = None


def do_ring_attention(
        q,
        k,
        v,
        head_num,
        softmax_scale,
        is_causal,
        fa_layout="SBH",
        attn_mask=None,
        dropout_p=0.,
        packed_seq_params=None,
):
    ps = ParallelState()
    cp_group = ps.get_group("cp")
    cp_size = ps.context_parallel_size
    rank = ps.get_rank("cp")
    cp_global_ranks = ps.get_device_mesh("cp").mesh.tolist()

    cp_para = dict()

    cp_para['causal'] = is_causal
    cp_para['cp_group'] = cp_group
    cp_para['cp_size'] = cp_size
    cp_para['rank'] = rank

    cp_para['cp_global_ranks'] = cp_global_ranks
    cp_para['cp_group_for_send_recv_overlap'] = None
    cp_para['megatron_cp_in_bnsd'] = fa_layout.upper() == "BNSD"

    if is_causal or fa_layout.upper() == "SBH" or fa_layout.upper() == "BNSD":

        output = ringattn_context_parallel(q, k, v, head_num, cp_para, softmax_scale, attn_mask, dropout_p,
                                           packed_seq_params=packed_seq_params)
    elif fa_layout.upper() == "TND":

        output = ringattn_context_parallel_tnd_general(q, k, v, head_num, cp_para, softmax_scale, attn_mask, dropout_p,
                                                       packed_seq_params=packed_seq_params)
    else:
        raise ValueError(f"Ring Attention only supports fa layout: `SBH`, `BNSD` and `TND`, but got {fa_layout.upper()}.")

    return output
