# coding=utf-8
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test DatasetAttr parsing, env helpers and dataset list building."""

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from mindspeed_llm.fsdp2.data.parser import (
    DATA_CONFIG,
    DatasetAttr,
    build_dataset_attr_from_dict,
    get_dataset_list,
    is_env_enabled,
    use_modelscope,
    use_openmind,
)
from tests.test_tools.utils import create_testconfig

_HUB_FUNCS = {"use_modelscope": use_modelscope, "use_openmind": use_openmind}


@pytest.fixture(autouse=True)
def _live_environ_getenv(monkeypatch):
    # datasets.Dataset.map(num_proc=...) can permanently rebind os.environ to a
    # deepcopy; if torch_npu previously wrapped os.environ.get with a logging
    # helper, the copy inherits an instance-level ``get`` still bound to the
    # stale environ, so monkeypatch.setenv values become invisible. Drop the
    # stale instance-level wrapper and point os.getenv at the live os.environ
    # so the env-driven branches below are tested against the real environment.
    if "get" in vars(os.environ):
        monkeypatch.delattr(os.environ, "get")
    monkeypatch.setattr(os, "getenv", lambda key, default=None: os.environ.get(key, default))


class TestDatasetAttrDefaults:
    """DatasetAttr default field values."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, load_from, dataset_name, expected", test_config["test_dataset_attr_defaults"])
    def test_dataset_attr_defaults(self, label, load_from, dataset_name, expected):
        attr = DatasetAttr(load_from=load_from, dataset_name=dataset_name)
        for field_name, value in expected.items():
            assert getattr(attr, field_name) == value

    def test_repr_returns_dataset_name(self):
        attr = DatasetAttr(load_from="hf_hub", dataset_name="org/name")
        assert repr(attr) == "org/name"


class TestSetAttr:
    """DatasetAttr.set_attr config lookup and default handling."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, key, config, use_default, default, expected", test_config["test_set_attr"])
    def test_set_attr(self, label, key, config, use_default, default, expected):
        attr = DatasetAttr(load_from="file", dataset_name="data.json")
        if use_default:
            attr.set_attr(key, config, default=default)
        else:
            attr.set_attr(key, config)
        assert getattr(attr, key) == expected


class TestJoin:
    """DatasetAttr.join merges config into fields, resetting absent keys."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, config, expected", test_config["test_join"])
    def test_join(self, label, config, expected):
        attr = DatasetAttr(load_from="file", dataset_name="data.json")
        attr.join(config)
        for field_name, value in expected.items():
            assert getattr(attr, field_name) == value


class TestEnvHelpers:
    """is_env_enabled and hub-selection env flags."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, value, use_default, default, expected", test_config["test_is_env_enabled"])
    def test_is_env_enabled(self, label, value, use_default, default, expected, monkeypatch):
        if value is None:
            monkeypatch.delenv("WCA_UT_FAKE_FLAG", raising=False)
        else:
            monkeypatch.setenv("WCA_UT_FAKE_FLAG", value)
        if use_default:
            assert is_env_enabled("WCA_UT_FAKE_FLAG", default=default) is expected
        else:
            assert is_env_enabled("WCA_UT_FAKE_FLAG") is expected

    @pytest.mark.parametrize("label, func, env_name, env_value, expected", test_config["test_hub_flag"])
    def test_hub_flag(self, label, func, env_name, env_value, expected, monkeypatch):
        if env_value is None:
            monkeypatch.delenv(env_name, raising=False)
        else:
            monkeypatch.setenv(env_name, env_value)
        assert _HUB_FUNCS[func]() is expected


class TestGetDatasetList:
    """get_dataset_list name resolution across dict/local/online/remote sources."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, dataset, dataset_dir, expected_names, expected_load_from", test_config["test_get_dataset_list"]
    )
    def test_get_dataset_list(self, label, dataset, dataset_dir, expected_names, expected_load_from):
        attrs = get_dataset_list(dataset, dataset_dir)
        assert [attr.dataset_name for attr in attrs] == expected_names
        if expected_load_from is not None:
            assert attrs[0].load_from == expected_load_from

    @pytest.mark.parametrize("label, ms_env, om_env, expected_load_from", test_config["test_online_hub"])
    def test_online_hub(self, label, ms_env, om_env, expected_load_from, monkeypatch):
        for env_name, env_value in (("USE_MODELSCOPE_HUB", ms_env), ("USE_OPENMIND_HUB", om_env)):
            if env_value is None:
                monkeypatch.delenv(env_name, raising=False)
            else:
                monkeypatch.setenv(env_name, env_value)
        attrs = get_dataset_list("org/name", "ONLINE")
        assert len(attrs) == 1
        assert attrs[0].load_from == expected_load_from

    def test_invalid_input_type_raises(self):
        with pytest.raises(ValueError, match="Invalid dataset input"):
            get_dataset_list(123, "unused_dir")

    def test_undefined_name_raises(self, tmp_path):
        (tmp_path / DATA_CONFIG).write_text(json.dumps({"alpaca": {"file_name": "a.json"}}), encoding="utf-8")
        with pytest.raises(ValueError, match="Undefined dataset 'missing'"):
            get_dataset_list("missing", str(tmp_path))

    def test_local_dataset_dir_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Cannot open"):
            get_dataset_list("alpaca", str(tmp_path))

    def test_local_dataset_dir_reads_dataset_info_json(self, tmp_path):
        (tmp_path / DATA_CONFIG).write_text(
            json.dumps({"alpaca": {"file_name": "a.json", "ranking": True}}), encoding="utf-8"
        )
        attrs = get_dataset_list("alpaca", str(tmp_path))
        assert len(attrs) == 1
        assert attrs[0].load_from == "file"
        assert attrs[0].dataset_name == "a.json"
        assert attrs[0].ranking is True

    def test_remote_prefix_downloads_via_hf_hub(self, tmp_path, monkeypatch):
        config_path = tmp_path / DATA_CONFIG
        config_path.write_text(json.dumps({"alpaca": {"file_name": "a.json"}}), encoding="utf-8")
        recorded = {}
        with mock.patch(
            "mindspeed_llm.fsdp2.data.parser.hf_hub_download",
            side_effect=lambda **kw: recorded.update(kw) or str(config_path),
        ) as mocked:
            attrs = get_dataset_list("alpaca", "REMOTE:org/repo")
        mocked.assert_called_once()
        assert recorded == {"repo_id": "org/repo", "filename": DATA_CONFIG, "repo_type": "dataset"}
        assert attrs[0].dataset_name == "a.json"

    def test_remote_download_failure_propagates(self, monkeypatch):
        # the download call sits outside the try block, so hub errors propagate unwrapped
        with mock.patch("mindspeed_llm.fsdp2.data.parser.hf_hub_download", side_effect=OSError("network down")):
            with pytest.raises(OSError, match="network down"):
                get_dataset_list("alpaca", "REMOTE:org/repo")


class TestBuildDatasetAttrFromDict:
    """build_dataset_attr_from_dict source-key dispatch and field mapping."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, config, expected", test_config["test_build_dataset_attr_from_dict"])
    def test_build_dataset_attr_from_dict(self, label, config, expected):
        if expected is None:
            with pytest.raises(ValueError, match="Config must contain one of"):
                build_dataset_attr_from_dict(config)
            return
        attr = build_dataset_attr_from_dict(config)
        for field_name, value in expected.items():
            assert getattr(attr, field_name) == value
