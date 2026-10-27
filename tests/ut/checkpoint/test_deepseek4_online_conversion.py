# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest

from mindspeed_llm.tasks.checkpoint import convert_ckpt_deepseek4
from mindspeed_llm.training import checkpointing


def _recording_converter(events):
    class RecordingConverter:
        def __init__(self, args, from_train=False):
            events.append(("init", args, from_train))

        def run(self):
            events.append(("run",))

    return RecordingConverter


@pytest.fixture
def rank_zero(monkeypatch):
    monkeypatch.setattr(checkpointing.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(checkpointing.torch.distributed, "barrier", lambda: None)
    monkeypatch.setenv("LOCAL_RANK", "0")


@pytest.mark.parametrize("model_type", ["deepseek4", "deepseek4_base"])
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.usefixtures("rank_zero")
def test_deepseek4_hf2mg_uses_training_converter(monkeypatch, model_type, shared):
    """Check HF-to-MG routing and from_train forwarding with mock converters."""
    events = []
    args = SimpleNamespace(model_type_hf=model_type)

    monkeypatch.setattr(
        checkpointing,
        "DeepSeek4Hf2MgConvert",
        _recording_converter(events),
    )

    checkpointing._convert_weights_if_needed(args, shared)

    assert events == [("init", args, True), ("run",)]


@pytest.mark.parametrize("model_type", ["deepseek4", "deepseek4_base"])
@pytest.mark.usefixtures("rank_zero")
def test_deepseek4_mg2hf_uses_training_converter(monkeypatch, tmp_path, model_type):
    """Check MG-to-HF routing, output path and from_train with mock converters."""
    events = []
    args = SimpleNamespace(
        model_type_hf=model_type,
        save=str(tmp_path / "mg"),
        hf_save_dir=str(tmp_path / "hf"),
    )

    monkeypatch.setattr(
        checkpointing,
        "DeepSeek4Mg2hfConvert",
        _recording_converter(events),
    )

    checkpointing._convert_weights_mg2hf(args, iteration=7)

    assert args.hf_save_dir == str(tmp_path / "hf" / "mg2hf_iteration7")
    assert events == [("init", args, True), ("run",)]


@pytest.mark.parametrize(
    ("converter_class", "base_class"),
    [
        (
            convert_ckpt_deepseek4.DeepSeek4Hf2MgConvert,
            convert_ckpt_deepseek4.Hf2MgConvert,
        ),
        (
            convert_ckpt_deepseek4.DeepSeek4Mg2hfConvert,
            convert_ckpt_deepseek4.Mg2HfConvert,
        ),
    ],
)
def test_deepseek4_converter_forwards_from_train(monkeypatch, converter_class, base_class):
    """Check argument forwarding to mock base constructors."""
    args = SimpleNamespace()

    def capture_base_init(self, init_args, from_train=False):
        assert init_args is args
        assert from_train is True
        raise RuntimeError("base constructor reached")

    monkeypatch.setattr(base_class, "__init__", capture_base_init)

    with pytest.raises(RuntimeError, match="base constructor reached"):
        converter_class(args, from_train=True)
