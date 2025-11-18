# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
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

import sys
import os
import shutil
from pathlib import Path
from tests.test_tools.utils import create_testconfig, weight_compare, run_cmd, weight_compare_hash


BASE_DIR = Path(__file__).absolute().parents[3]
CKPT_PYPATH = os.path.join(BASE_DIR, "mindspeed_llm/tasks/checkpoint/convert_ckpt_mamba2.py")


class TestCheckpoint(object):
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))
    test_config_cmd = create_testconfig(Path(__file__).with_suffix(".json"), cmd=True)

    def test_mamba2_hf2mcore_tp1pp2(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_mamba2_hf2mcore_tp1pp2'])
        assert exit_code == 0
        base_hash = self.test_config['test_mamba2_hf2mcore_tp1pp2'][1]
        save_dir = self.test_config['test_mamba2_hf2mcore_tp1pp2'][0]['save-dir']
        assert weight_compare_hash(save_dir, base_hash, "pt")

    def test_mamba2_mcore2hf_tp1pp2(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        base_hash = self.test_config['test_mamba2_mcore2hf_tp1pp2'][1]
        load_dir = self.test_config['test_mamba2_mcore2hf_tp1pp2'][0]['load-dir']        
        save_dir = os.path.join(self.test_config['test_mamba2_mcore2hf_tp1pp2'][0]['save-dir'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_mamba2_mcore2hf_tp1pp2'])
        assert exit_code == 0
        assert weight_compare_hash(save_dir, base_hash, "bin")
        shutil.rmtree(load_dir)
        shutil.rmtree(save_dir)
