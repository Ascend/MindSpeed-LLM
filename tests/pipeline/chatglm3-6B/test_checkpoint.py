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
"""Test of checkpoint conversion"""

import sys
import os
import shutil
from pathlib import Path
import logging
import re
import math
import pytest
import mindspeed_llm
from tests.test_tools.utils import create_testconfig, weight_compare, run_cmd


BASE_DIR = Path(__file__).absolute().parents[3]
CKPT_PYPATH = os.path.join(BASE_DIR, "convert_ckpt.py")


class TestCheckpoint(object):
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))
    test_config_cmd = create_testconfig(Path(__file__).with_suffix(".json"), cmd=True)

    def test_chatglm3_hf2legacy_tp2pp4(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_chatglm3_hf2legacy_tp2pp4'])
        assert exit_code == 0
        base_dir = '/data/chatglm3-6b-base-mg-tp2pp4-legacy-base/'
        save_dir = self.test_config['test_chatglm3_hf2legacy_tp2pp4'][0]['save-dir']
        assert weight_compare(base_dir, save_dir)
        shutil.rmtree(save_dir)