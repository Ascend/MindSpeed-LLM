# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION. All rights reserved.
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
"""Tests of Checkpoint"""

import os
import shutil
import sys
from pathlib import Path

import pytest
import torch

from tests.test_tools.utils import create_testconfig, load_safetensors_to_dict, run_cmd, weight_compare_hash


BASE_DIR = Path(__file__).absolute().parents[3]
CKPT_PYPATH = os.path.join(BASE_DIR, "convert_ckpt.py")
CKPTV2_PYPATH = os.path.join(BASE_DIR, "convert_ckpt_v2.py")


class TestCheckpoint:
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))
    test_config_cmd = create_testconfig(Path(__file__).with_suffix(".json"), cmd=True)

    def test_deepseek3_hf2mcore_tp2pp2vpp1ep2nooplayer(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(
            ["python3", CKPTV2_PYPATH] + self.test_config_cmd['test_deepseek3_hf2mcore_tp2pp2vpp1ep2nooplayer']
        )
        assert exit_code == 0
        base_hash = self.test_config['test_deepseek3_hf2mcore_tp2pp2vpp1ep2nooplayer'][1]
        save_dir = self.test_config['test_deepseek3_hf2mcore_tp2pp2vpp1ep2nooplayer'][0]['save-dir']
        assert weight_compare_hash(save_dir, base_hash, "pt")
        shutil.rmtree(save_dir)

    @pytest.mark.parametrize("params", test_config["test_deepseek4_flash_l1_bf16_roundtrip"])
    def test_deepseek4_flash_l1_bf16_roundtrip(self, params, tmp_path):
        """Check real single-layer HF-to-MG-to-HF conversion."""
        load_dir = params["load-dir"]
        mg_dir = str(tmp_path / "mg")
        hf_dir = str(tmp_path / "hf")
        for source, target, kind in [(load_dir, mg_dir, "hf"), (mg_dir, hf_dir, "mg")]:
            command = [sys.executable, CKPTV2_PYPATH, "--model-type-hf", "deepseek4"]
            command += ["--expert-tensor-parallel-size", "1"]
            command += ["--load-model-type", kind, "--load-dir", source, "--save-dir", target]
            command += ["--save-model-type", "mg" if kind == "hf" else "hf"]
            assert run_cmd(command) == 0
        expected = load_safetensors_to_dict(load_dir)
        actual = load_safetensors_to_dict(hf_dir)
        assert expected
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
