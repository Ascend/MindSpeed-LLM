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

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional


@dataclass
class InstructionDatasetAttr:
    r"""
    Dataset attributes.
    """

    """ basic configs """
    load_from: Literal["file"]
    dataset_name: str
    """ extra configs """
    subset: Optional[str] = None
    folder: Optional[str] = None
    ranking: bool = False
    formatting: Literal["alpaca", "sharegpt", "openai"] = "alpaca"
    """ columns """
    system: Optional[str] = None
    images: Optional[str] = None
    """ columns for the alpaca format """
    prompt: Optional[str] = "instruction"
    query: Optional[str] = "input"
    response: Optional[str] = "output"
    history: Optional[str] = None
    """ columns for the sharegpt format """
    messages: Optional[str] = "conversations"
    tools: Optional[str] = None
    """ columns for the pairwise dataset """
    chosen: Optional[str] = "chosen"
    rejected: Optional[str] = "rejected"
    """ tags for the sharegpt format """
    role_tag: Optional[str] = "from"
    content_tag: Optional[str] = "value"
    user_tag: Optional[str] = "human"
    assistant_tag: Optional[str] = "gpt"
    observation_tag: Optional[str] = "observation"
    function_tag: Optional[str] = "function_call"
    system_tag: Optional[str] = "system"

    def set_attr(self, key: str, obj: Dict[str, Any], default: Optional[Any] = None) -> None:
        setattr(self, key, obj.get(key, default))