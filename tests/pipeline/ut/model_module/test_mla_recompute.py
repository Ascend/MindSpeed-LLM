from types import SimpleNamespace

import pytest

from mindspeed_llm.features_manager.transformer.multi_latent_attention.mla_feature import MLAFeature
from mindspeed_llm.tasks.models.transformer import mla_up_proj_overlap_tp_comm


def _mla_args(tp_size):
    return SimpleNamespace(
        multi_latent_attention=True,
        kv_lora_rank=512,
        v_head_dim=128,
        qk_pos_emb_head_dim=64,
        qk_head_dim=128,
        padded_base_length=128,
        mla_up_proj_tp_overlap=False,
        sequence_parallel=False,
        recompute_mla_up_proj=True,
        tensor_model_parallel_size=tp_size,
        mla_zero_memory=False,
        mla_swap_core_attn_out=False,
    )


def test_recompute_mla_up_proj_allows_tp1_without_tp_overlap():
    MLAFeature().validate_args(_mla_args(tp_size=1))


def test_recompute_mla_up_proj_rejects_tp2_without_tp_overlap():
    with pytest.raises(ValueError, match="tensor model parallel size is 1"):
        MLAFeature().validate_args(_mla_args(tp_size=2))


@pytest.mark.parametrize(
    ("schedules_method", "post_process", "expected"),
    [
        (None, False, True),
        ("dualpipev", False, True),
        ("dualpipev", True, False),
    ],
)
def test_should_recompute_mla_up_proj_respects_dualpipev_post_process(
    monkeypatch, schedules_method, post_process, expected
):
    monkeypatch.setattr(mla_up_proj_overlap_tp_comm, "get_post_process_flag", lambda: post_process)
    args = SimpleNamespace(schedules_method=schedules_method)

    assert mla_up_proj_overlap_tp_comm.should_recompute_mla_up_proj(args, True) is expected
