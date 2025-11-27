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
CKPT_PYPATH = os.path.join(BASE_DIR, "mindspeed_llm/tasks/checkpoint/convert_param.py")


class TestCheckpoint(object):
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))
    test_config_cmd = create_testconfig(Path(__file__).with_suffix(".json"), cmd=True)

    def test_llama3_hf2mcore_tp8pp1_tp2d(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_llama3_hf2mcore_tp8pp1_tp2d'])
        assert exit_code == 0
        base_hash = self.test_config['test_llama3_hf2mcore_tp8pp1_tp2d'][1]
        save_dir = self.test_config['test_llama3_hf2mcore_tp8pp1_tp2d'][0]['mg-dir']
        assert weight_compare_hash(save_dir, base_hash, "pt")

    def test_llama3_mcore2hf_tp8pp1_tp2d(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        base_hash = self.test_config['test_llama3_mcore2hf_tp8pp1_tp2d'][1]
        load_dir = self.test_config['test_llama3_mcore2hf_tp8pp1_tp2d'][0]['mg-dir']        
        save_dir = os.path.join(self.test_config['test_llama3_mcore2hf_tp8pp1_tp2d'][0]['hf-dir'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_llama3_mcore2hf_tp8pp1_tp2d'])
        assert exit_code == 0
        assert weight_compare_hash(save_dir, base_hash, "safetensors")
        shutil.rmtree(load_dir)
        shutil.rmtree(save_dir)
