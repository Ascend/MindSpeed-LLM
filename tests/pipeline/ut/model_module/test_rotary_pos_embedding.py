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
"""Test of Rotary Pos Embedding"""
from types import SimpleNamespace
from pathlib import Path
import pytest
import torch
from mindspeed_llm import megatron_adaptor
from tests.test_tools.dist_test import create_testconfig
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding


class TestRotaryPosEmbedding:
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.fixture
    def mock_dependency(self, request):
        # init test name space
        def get_test_namespace():
            test_name_space = SimpleNamespace()
            test_name_space.use_glm_rope = request.getfixturevalue("chatglm")
            test_name_space.rope_scaling_type = None
            test_name_space.rotary_base = request.getfixturevalue("rotary_base")
            test_name_space.tp_2d = False
            test_name_space.tp_x = 1
            test_name_space.tp_y = 1
            test_name_space.dynamic_factor = 1.0
            test_name_space.neat_pack = False
            test_name_space.partial_rotary_factor = False
            return test_name_space
        # set up name space function
        import mindspeed_llm
        setattr(mindspeed_llm.core.models.common.embeddings.rotary_pos_embedding, "get_args", get_test_namespace)

    @pytest.mark.parametrize("rotary_param, chatglm, rotary_base, seq, position_ids, packed_seq, expected", test_config["test_rotary_pos_embedding"])
    def test_rotary_pos_embedding(self, mock_dependency, rotary_param, chatglm, rotary_base, seq, position_ids, packed_seq, expected):
        rotary = RotaryEmbedding(**rotary_param)
        assert(torch.allclose(rotary.forward(seq, position_ids, packed_seq).cpu(), torch.Tensor(expected)))
        