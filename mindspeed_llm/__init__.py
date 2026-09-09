# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
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
import os
import sys

llm_version = os.environ.get("MINDSPEED_LLM_VERSION", "012").lower()
if llm_version == "018":
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(_repo_root, "tests"))
    sys.modules.pop(__name__, None)
    _mindspeed_llm = __import__(__name__)
    sys.modules[__name__] = _mindspeed_llm
    globals().update(_mindspeed_llm.__dict__)
else:
    backend = os.environ.get("TRAINING_BACKEND", "mcore").lower()
    if backend == "mcore":
        from mindspeed_llm.tasks import megatron_adaptor_v2 as megatron_adaptor  # noqa: F401
