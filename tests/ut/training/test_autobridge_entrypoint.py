# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU-only tests of the AutoBridge entrypoint's training API contract."""

import argparse
from contextlib import contextmanager
import importlib.util
import os
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = REPO_ROOT / "pretrain_autobridge.py"


@pytest.fixture(autouse=True)
def live_environ_get(monkeypatch):
    """Read current environment values despite a copied instance-level wrapper."""
    if "get" in vars(os.environ):
        monkeypatch.delattr(os.environ, "get")


@pytest.fixture(name="entrypoint_runtime")
def fake_entrypoint_runtime(monkeypatch):
    """Replace the NPU runtime boundaries while executing the real entrypoint."""
    events = []
    config = object()
    model_type = object()

    def datasets(*_):
        return None

    def step(*_):
        return None

    def embedding_ranks(ranks):
        return ranks

    spec = importlib.util.spec_from_file_location(
        "autobridge_entrypoint_under_test", REPO_ROOT / "tests/mindspeed_llm/training/autobridge.py"
    )
    autobridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(autobridge)
    monkeypatch.setattr(autobridge, "install_autobridge_feature", lambda: events.append("install"))

    compat_spec = importlib.util.spec_from_file_location(
        "autobridge_compat_under_test", REPO_ROOT / "tests/mindspeed_llm/training/autobridge_compat.py"
    )
    compat = importlib.util.module_from_spec(compat_spec)
    compat_spec.loader.exec_module(compat)

    @contextmanager
    def training_scope():
        events.append("scope_enter")
        try:
            yield
        finally:
            events.append("scope_exit")

    monkeypatch.setattr(compat, "autobridge_training_scope", training_scope)

    def parse_and_validate_args(*, extra_args_provider, args_defaults):
        assert events == ["install", "scope_enter"]
        parser = argparse.ArgumentParser()
        parser.add_argument("--tokenizer-type", default=args_defaults["tokenizer_type"])
        extra_args_provider(parser)
        args = parser.parse_args()
        events.append("parse")
        return args

    def make_config(args):
        assert events == ["install", "scope_enter", "parse"]
        assert args.pretrained_model_path == "/models/qwen35"
        assert args.tokenizer_type == "GPT2BPETokenizer"
        events.append("config")
        return config

    def pretrain(cfg_container, dataset_provider, model_provider, kind, forward_step, *, get_embedding_ranks):
        assert events == ["install", "scope_enter", "parse", "config"]
        assert cfg_container is config
        assert dataset_provider is datasets and dataset_provider.is_distributed
        assert model_provider is autobridge.autobridge_model_provider
        assert kind is model_type
        assert forward_step is step
        assert get_embedding_ranks is embedding_ranks
        events.append("train")

    modules = {
        "megatron.core.enums": {"ModelType": SimpleNamespace(encoder_or_decoder=model_type)},
        "megatron.core.process_groups_config": {"ProcessGroupCollection": object},
        "megatron.training.arguments": {"parse_and_validate_args": parse_and_validate_args},
        "megatron.training.argument_utils": {"pretrain_cfg_container_from_args": make_config},
        "mindspeed_llm.training.training": {"pretrain": pretrain},
        "mindspeed_llm.training.utils": {"auto_coverage": lambda function: function},
        "pretrain_gpt": {
            "forward_step": step,
            "get_embedding_ranks": embedding_ranks,
            "train_valid_test_datasets_provider": datasets,
        },
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "mindspeed_llm.training.autobridge", autobridge)
    monkeypatch.setitem(sys.modules, "mindspeed_llm.training.autobridge_compat", compat)
    monkeypatch.setenv("MINDSPEED_LLM_VERSION", "018")
    # The production entrypoint prepends its source tree; restore sys.path after the test.
    monkeypatch.setattr(sys, "path", list(sys.path))
    return events


def test_entrypoint_builds_config_before_calling_current_pretrain(entrypoint_runtime, monkeypatch):
    """
    Author: tichang
    Date: 2026-10-20
    Description: Build configuration before training within the AutoBridge compatibility scope.
    Remarks: CPU-only; no Bridge repositories, NPU, model weights or datasets are required.
    """
    monkeypatch.setattr(sys, "argv", [str(ENTRYPOINT), "--pretrained-model-path", "/models/qwen35"])

    runpy.run_path(str(ENTRYPOINT), run_name="__main__")

    assert entrypoint_runtime == ["install", "scope_enter", "parse", "config", "train", "scope_exit"]


def test_entrypoint_rejects_missing_hf_path_before_training(entrypoint_runtime, monkeypatch):
    """
    Author: tichang
    Date: 2026-10-20
    Description: Reject a missing HF path and leave the compatibility scope before training.
    Remarks: CPU-only; no Bridge repositories, NPU, model weights or datasets are required.
    """
    monkeypatch.setattr(sys, "argv", [str(ENTRYPOINT)])

    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(ENTRYPOINT), run_name="__main__")

    assert error.value.code == 2
    assert entrypoint_runtime == ["install", "scope_enter", "scope_exit"]


def test_entrypoint_rejects_non_018_runtime(monkeypatch):
    """
    Author: tichang
    Date: 2026-10-20
    Description: Reject a runtime version other than the supported 018 entrypoint.
    Remarks: CPU-only; no Bridge repositories, NPU, model weights or datasets are required.
    """
    monkeypatch.setenv("MINDSPEED_LLM_VERSION", "012")

    with pytest.raises(RuntimeError, match="MINDSPEED_LLM_VERSION=018"):
        runpy.run_path(str(ENTRYPOINT), run_name="__main__")
