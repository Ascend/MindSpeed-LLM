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

import sys
import os
import shutil
from pathlib import Path
import logging
import re
import math
import pytest
from mindspeed_llm import megatron_adaptor
from tests.test_tools.utils import create_testconfig, weight_compare, run_cmd, weight_compare_hash


BASE_DIR = Path(__file__).absolute().parents[3]
CKPT_PYPATH = os.path.join(BASE_DIR, "convert_ckpt.py")


class TestCheckpoint(object):
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))
    test_config_cmd = create_testconfig(Path(__file__).with_suffix(".json"), cmd=True)


    def test_mixtral_hf2mcore_tp2pp2ep2dypp(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_mixtral_hf2mcore_tp2pp2ep2dypp'])
        assert exit_code == 0
        base_hash = self.test_config['test_mixtral_hf2mcore_tp2pp2ep2dypp'][1]
        save_dir = self.test_config['test_mixtral_hf2mcore_tp2pp2ep2dypp'][0]['save-dir']
        assert weight_compare_hash(save_dir, base_hash, "pt")
        shutil.rmtree(save_dir)


    def test_mixtral_mcore2hf_tp1pp4ep2vpp2(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_mixtral_mcore2hf_tp1pp4ep2vpp2'])
        assert exit_code == 0
        base_hash = self.test_config['test_mixtral_mcore2hf_tp1pp4ep2vpp2'][1]
        save_dir = os.path.join(self.test_config['test_mixtral_mcore2hf_tp1pp4ep2vpp2'][0]['save-dir'], 'mg2hf')
        assert weight_compare_hash(save_dir, base_hash, "safetensors")
        shutil.rmtree(save_dir)


    def test_deepseek2_lite_hf2mcore_tp1pp1ep8(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_deepseek2_lite_hf2mcore_tp1pp1ep8'])
        assert exit_code == 0
        base_hash = self.test_config['test_deepseek2_lite_hf2mcore_tp1pp1ep8'][1]
        save_dir = self.test_config['test_deepseek2_lite_hf2mcore_tp1pp1ep8'][0]['save-dir']
        assert weight_compare_hash(save_dir, base_hash, "pt")
        shutil.rmtree(save_dir)


    def test_deepseek2_lite_mcore2hf_tp1pp1ep8(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_deepseek2_lite_mcore2hf_tp1pp1ep8'])
        assert exit_code == 0
        base_hash = self.test_config['test_deepseek2_lite_mcore2hf_tp1pp1ep8'][1]
        save_dir = os.path.join(self.test_config['test_deepseek2_lite_mcore2hf_tp1pp1ep8'][0]['save-dir'], 'mg2hf')
        assert weight_compare_hash(save_dir, base_hash, "safetensors")
        shutil.rmtree(save_dir)


    def test_gemma2_hf2mcore_tp8pp1(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_gemma2_hf2mcore_tp8pp1'])
        assert exit_code == 0
        base_hash = self.test_config['test_gemma2_hf2mcore_tp8pp1'][1]
        save_dir = self.test_config['test_gemma2_hf2mcore_tp8pp1'][0]['save-dir']
        assert weight_compare_hash(save_dir, base_hash, "pt")
        shutil.rmtree(save_dir)


    def test_llama2_lora2mcore_tp1pp1(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_llama2_lora2mcore_tp1pp1'])
        assert exit_code == 0
        base_hash = self.test_config['test_llama2_lora2mcore_tp1pp1'][1]
        save_dir = self.test_config['test_llama2_lora2mcore_tp1pp1'][0]['save-dir']
        assert weight_compare_hash(save_dir, base_hash, "pt")
        shutil.rmtree(save_dir)


    def test_qwen2_moe_hf2mcore_tp1pp2ep2(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_qwen2_moe_hf2mcore_tp1pp2ep2'])
        assert exit_code == 0
        base_hash = self.test_config['test_qwen2_moe_hf2mcore_tp1pp2ep2'][1]
        save_dir = self.test_config['test_qwen2_moe_hf2mcore_tp1pp2ep2'][0]['save-dir']
        assert weight_compare_hash(save_dir, base_hash, "pt")


    def test_qwen2_moe_mcore2hf_tp1pp2ep2(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_qwen2_moe_mcore2hf_tp1pp2ep2'])
        assert exit_code == 0
        base_hash = self.test_config['test_qwen2_moe_mcore2hf_tp1pp2ep2'][1]
        load_dir = self.test_config['test_qwen2_moe_mcore2hf_tp1pp2ep2'][0]['load-dir']
        save_dir = os.path.join(self.test_config['test_qwen2_moe_mcore2hf_tp1pp2ep2'][0]['save-dir'], 'mg2hf')
        assert weight_compare_hash(save_dir, base_hash, "safetensors")
        shutil.rmtree(load_dir)
        shutil.rmtree(save_dir)


    def test_qwen2_moe_mcore2hf_tp2pp1ep2(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_qwen2_moe_mcore2hf_tp2pp1ep2'])
        assert exit_code == 0
        base_hash = self.test_config['test_qwen2_moe_mcore2hf_tp2pp1ep2'][1]
        save_dir = os.path.join(self.test_config['test_qwen2_moe_mcore2hf_tp2pp1ep2'][0]['save-dir'], 'mg2hf')
        assert weight_compare_hash(save_dir, base_hash, "safetensors")
        shutil.rmtree(save_dir)


    def test_llama3_noop_layer_hf2mg(self):
        """
        Test case for nooplayer.
        """
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_llama3_noop_layer_hf2mg'])
        assert exit_code == 0
        base_hash = self.test_config['test_llama3_noop_layer_hf2mg'][1]
        save_dir = self.test_config['test_llama3_noop_layer_hf2mg'][0]['save-dir']
        assert weight_compare_hash(save_dir, base_hash, "pt")
        shutil.rmtree(save_dir)


    def test_qwen2_hf2mcore_tp1pp1(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_qwen2_hf2mcore_tp1pp1'])
        assert exit_code == 0
        base_hash = self.test_config['test_qwen2_hf2mcore_tp1pp1'][1]
        save_dir = self.test_config['test_qwen2_hf2mcore_tp1pp1'][0]['save-dir']
        assert weight_compare_hash(save_dir, base_hash, "pt")
        shutil.rmtree(save_dir)


    def test_qwen2_moe_hf2mcore_tp2pp1ep2_extend(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_qwen2_moe_hf2mcore_tp2pp1ep2_extend'])
        assert exit_code == 0
        base_hash = self.test_config['test_qwen2_moe_hf2mcore_tp2pp1ep2_extend'][1]
        save_dir = self.test_config['test_qwen2_moe_hf2mcore_tp2pp1ep2_extend'][0]['save-dir']
        assert weight_compare_hash(save_dir, base_hash, "pt")
        shutil.rmtree(save_dir)


    def test_qwen2_moe_hf2mcore_tp2pp1ep2(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_qwen2_moe_hf2mcore_tp2pp1ep2'])
        assert exit_code == 0
        base_dir = "/data/ci/models/qwen2-moe/mg/qwen2_moe_l2_t2p1e2_base"
        save_dir = self.test_config['test_qwen2_moe_hf2mcore_tp2pp1ep2'][0]['save-dir']
        assert weight_compare(base_dir, save_dir)
        shutil.rmtree(save_dir)