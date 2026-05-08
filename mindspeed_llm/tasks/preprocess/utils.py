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

import bisect
import os
import json
import logging
from functools import partial
from typing import Any, Dict, List, Sequence
from datasets import load_dataset, concatenate_datasets, interleave_datasets

from mindspeed_llm.tasks.preprocess.templates import Role
from mindspeed_llm.tasks.preprocess.parser import InstructionDatasetAttr


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LlamaFactoryInstructionHandler preprocess data supported format
FILEEXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}

DATA_CONFIG = "dataset_info.json"


def check_dataset_info_map(data_args, column_names, raw_datasets, tag_names=None):
    if len(data_args.map_keys.keys()) > len(column_names):
        raise ValueError("Please check map_keys")

    for key in data_args.map_keys.keys():
        if key not in column_names:
            raise ValueError(f' {key} is unvalid, Please check map_keys')

    if "AlpacaStyle" in data_args.handler_name:
        for value in data_args.map_keys.values():
            if value and value not in raw_datasets.format['columns']:
                raise ValueError(f' {value} is unvalid, Please check map_keys')

    if "SharegptStyle" in data_args.handler_name:
        if "tags" in data_args.map_keys.keys():
            for tag_name in data_args.map_keys["tags"].keys():
                if tag_name not in tag_names:
                    raise ValueError(f'tag_name {tag_name} is unvalid, Please check map_keys')


def get_handler_dataset_attr(data_args, raw_datasets):
    dataset_attr = InstructionDatasetAttr("file", dataset_name=data_args.handler_name)
    dataset_attr.dataset_additional_keys = data_args.dataset_additional_keys

    if "Pairwise" in data_args.handler_name:
        setattr(dataset_attr, "ranking", True)

    if "AlpacaStyle" in data_args.handler_name:
        dataset_attr.formatting = "alpaca"
        column_names = ["prompt", "query", "response", "history", "system", "chosen", "rejected"]
        if data_args.map_keys is not None:
            check_dataset_info_map(data_args, column_names, raw_datasets, None)
            for column_name, target_name in data_args.map_keys.items():
                setattr(dataset_attr, column_name, target_name)

    elif "OpenAI" in data_args.handler_name:
        dataset_attr.formatting = "openai"
        openai_defaults = {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "observation_tag": "tool",
            "function_tag": "function_call",
            "system_tag": "system",
            "messages": "messages",
            "tools": "tools",
        }
        for k, v in openai_defaults.items():
            setattr(dataset_attr, k, v)

        column_names = ["messages", "system", "tools"]
        if data_args.map_keys is not None:
            check_dataset_info_map(data_args, column_names, raw_datasets, None)
            for column_name, target_name in data_args.map_keys.items():
                if column_name == "tags":
                    logger.warning(
                        "OpenAIInstructionHandler does not support custom 'tags' in --map-keys"
                    )
                    continue
                setattr(dataset_attr, column_name, target_name)

    elif "SharegptStyle" in data_args.handler_name or "Hunyuan" in data_args.handler_name:
        dataset_attr.formatting = "sharegpt"
        tag_names = ["role_tag", "content_tag", "user_tag", "assistant_tag", "observation_tag", "function_tag", "system_tag"]
        column_names = ["messages", "tags", "system", "tools", "chosen", "rejected", "kto_tag"]

        if data_args.map_keys is not None:
            check_dataset_info_map(data_args, column_names, raw_datasets, tag_names)
            for column_name, target_name in data_args.map_keys.items():
                if column_name == "tags":
                    for tag in tag_names:
                        dataset_attr.set_attr(tag, data_args.map_keys["tags"])
                else:
                    setattr(dataset_attr, column_name, target_name)

    return dataset_attr


def get_dataset_list(data_args) -> List["InstructionDatasetAttr"]:
    """
    Map multiple dataset attributes to List["InstructionDatasetAttr"]
    through parameters and the data.json mapping file.
    """
    if data_args.input is not None:
        dataset_names = [ds.split("/")[-1].strip() for ds in data_args.input.split(",")]
    else:
        dataset_names = []

    try:
        with open(os.path.join(data_args.dataset_dir, DATA_CONFIG), "r") as f:
            dataset_info = json.load(f)
    except Exception as err:
        if len(dataset_names) != 0:
            raise ValueError(
                "Cannot open {} due to {}.".format(os.path.join(data_args.dataset_dir, DATA_CONFIG), str(err))
            ) from err
        dataset_info = None

    if dataset_info is not None:
        raise ValueError(
                "Cannot load {}.".format(os.path.join(data_args.dataset_dir, DATA_CONFIG))
        )

    # Multiple Dataset Interleaving Probability
    if data_args.interleave_probs is not None:
        data_args.interleave_probs = [float(prob.strip()) for prob in data_args.interleave_probs.split(",")]

    dataset_list: List[InstructionDatasetAttr] = []
    for name in dataset_names:
        if name not in dataset_info:
            raise ValueError("Undefined dataset {} in {}.".format(name, DATA_CONFIG))

        if "script_url" in dataset_info[name]:
            dataset_attr = InstructionDatasetAttr("script", dataset_name=dataset_info[name]["script_url"])
        else:
            dataset_attr = InstructionDatasetAttr("file", dataset_name=dataset_info[name]["file_name"])

        dataset_attr.set_attr("subset", dataset_info[name])
        dataset_attr.set_attr("folder", dataset_info[name])
        dataset_attr.set_attr("ranking", dataset_info[name], default=False)
        dataset_attr.set_attr("formatting", dataset_info[name], default="alpaca")

        if "columns" in dataset_info[name]:
            column_names = ["system", "images"]
            if dataset_attr.formatting == "alpaca":
                column_names.extend(["prompt", "query", "response", "history"])
            elif dataset_attr.formatting == "openai":
                column_names.extend(["messages", "tools"])
            else:
                column_names.extend(["messages", "tools"])

            for column_name in column_names:
                dataset_attr.set_attr(column_name, dataset_info[name]["columns"])

        if dataset_attr.formatting == "sharegpt" and "tags" in dataset_info[name]:
            tag_names = (
                "role_tag",
                "content_tag",
                "user_tag",
                "assistant_tag",
                "observation_tag",
                "function_tag",
                "system_tag",
            )
            for tag in tag_names:
                dataset_attr.set_attr(tag, dataset_info[name]["tags"])

        dataset_list.append(dataset_attr)

    return dataset_list


def convert_alpaca_to_intermediate(sample: Dict[str, List[Any]], dataset_attr: "InstructionDatasetAttr"):
    """
    format sample info
    {
      "instruction": "我还想知道中国古代的五代十国时期和欧洲的中世纪有什么异同点？",
      "input": "",
      "output": "中国的五代十国时期和欧洲的中世纪大体上是同时期的历史时期，但它们有许多重要的异同点。",
      "history": [
       [
        "回答的非常好",
        "感谢你的认可！还有什么需要我帮助的吗？"
       ]
      ]
     }
    ---->>>>
    {
        'prompt': [{'role': 'user', 'content': '回答的非常好'}, 
                {'role': 'assistant', 'content': '感谢你的认可！还有什么需要我帮助的吗？'}, 
                {'role': 'user', 'content': '我还想知道中国古代的五代十国时期和欧洲的中世纪有什么异同点？'}], 
        'response': [{'role': 'assistant', 'content': '中国的五代十国时期和欧洲的中世纪大体上是同时期的历史时期，但它们有许多重要的异同点。'}], 
        'system': [''], 
        'tools': ['']
    }
    """
    outputs = {"prompt": [], "response": [], "system": [], "tools": []}
    prompt = []
    
    if dataset_attr.history and (isinstance(sample[dataset_attr.history], list) or isinstance(sample[dataset_attr.history], dict)):
        for old_prompt, old_response in sample[dataset_attr.history]:
            prompt.append({"role": Role.USER.value, "content": old_prompt})
            prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

    content = []
    if dataset_attr.prompt and sample[dataset_attr.prompt]:
        content.append(sample[dataset_attr.prompt])

    if dataset_attr.query and sample[dataset_attr.query]:
        content.append(sample[dataset_attr.query])

    prompt.append({"role": Role.USER.value, "content": "\n".join(content)})

    if dataset_attr.ranking:
        if dataset_attr.chosen and isinstance(sample[dataset_attr.chosen], list):
            response = [
                {"role": Role.ASSISTANT.value, "content": sample[dataset_attr.chosen][0]},
                {"role": Role.ASSISTANT.value, "content": sample[dataset_attr.rejected][1]},
            ]
        elif dataset_attr.chosen and isinstance(sample[dataset_attr.chosen], str):
            response = [
                {"role": Role.ASSISTANT.value, "content": sample[dataset_attr.chosen]},
                {"role": Role.ASSISTANT.value, "content": sample[dataset_attr.rejected]},
            ]
        else:
            response = []
    else:
        if dataset_attr.response and isinstance(sample[dataset_attr.response], list):
            response = [
                {
                    "role": Role.ASSISTANT.value,
                    "content": content
                }
                for content in sample[dataset_attr.response]
            ]
        elif dataset_attr.response and isinstance(sample[dataset_attr.response], str):
            response = [
                {
                    "role": Role.ASSISTANT.value, 
                    "content": sample[dataset_attr.response]
                }
            ]
        else:
            response = []

    outputs["prompt"] = prompt
    outputs["response"] = response
    outputs["system"].append(sample[dataset_attr.system] if dataset_attr.system else "")
    outputs["tools"].append("")

    for add_key in dataset_attr.dataset_additional_keys:
        if add_key != "labels":
            outputs[add_key] = sample[add_key]

    return outputs


def convert_sharegpt_to_intermediate(
    sample: Dict[str, List[Any]], dataset_attr: "InstructionDatasetAttr"):
    """
    convert sharegpt or openAI sharegpt to intermediate format
    sharegpt:
    [
    {
        "conversations": [
        {
            "from": "human",
            "value": ""
        },
        {
            "from": "function_call",
            "value": ""
        },
        {
            "from": "observation",
            "value": ""
        },
        {
            "from": "gpt",
            "value": ""
        }
        ],
        "system": "系统提示词（选填）",
        "tools": ""
    }
    ]

    ---->>>>

    {
        'prompt': [{'role': 'user', 'content': ''}, 
                {'role': 'assistant', 'content': ''}, 
                {'role': 'user', 'content': ''}], 
        'response': [{'role': 'assistant', 'content': ''}], 
        'system': [''], 
        'tools': ['']
    }
    """
    outputs = {"prompt": [], "response": [], "system": [], "tools": []}

    tag_mapping = {
        dataset_attr.user_tag: Role.USER.value,               # "human" : "user"
        dataset_attr.assistant_tag: Role.ASSISTANT.value,     # "gpt" : "assistant"
        dataset_attr.observation_tag: Role.OBSERVATION.value, # "observation" : "observation"
        dataset_attr.function_tag: Role.FUNCTION.value,       # "function_call" : "function"
        dataset_attr.system_tag: Role.SYSTEM.value,           # "system" : "system"
    }

    # "human" and "observation" must appear in odd-numbered positions
    # "gpt" and "function" must appear in even-numbered positions.
    odd_tags = (dataset_attr.user_tag, dataset_attr.observation_tag)
    even_tags = (dataset_attr.assistant_tag, dataset_attr.function_tag)
    accept_tags = (odd_tags, even_tags)

    messages = sample[dataset_attr.messages]
    if dataset_attr.system_tag and messages[0][dataset_attr.role_tag] == dataset_attr.system_tag:
        system = messages[0][dataset_attr.content_tag]
        messages = messages[1:]
    else:
        system = sample[dataset_attr.system] if dataset_attr.system else ""

    if len(messages) == 0:
        return outputs

    aligned_messages = []
    broken_data = False
    for turn_idx, message in enumerate(messages):
        if message[dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
            logger.warning("Invalid role tag in {}.".format(messages))
            broken_data = True

        content_value = message.get(dataset_attr.content_tag)

        if content_value is not None:
            aligned_messages.append(
                {"role": tag_mapping.get(message.get(dataset_attr.role_tag), "unknown"), "content": content_value}
            )
        else:
            logger.warning(f"Missing content tag in message at turn {turn_idx}: {message}")

    if (not dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
        dataset_attr.ranking and len(aligned_messages) % 2 == 0
    ):
        logger.warning("Invalid message count in {}.".format(messages))
        broken_data = True

    elif (
            dataset_attr.ranking
            and isinstance(sample[dataset_attr.chosen], dict)
            and isinstance(sample[dataset_attr.rejected], dict)
    ):  # pairwise example
        chosen = sample[dataset_attr.chosen]
        rejected = sample[dataset_attr.rejected]
        if (
                chosen[dataset_attr.role_tag] not in accept_tags[-1]
                or rejected[dataset_attr.role_tag] not in accept_tags[-1]
        ):
            logger.warning("Invalid role tag in {}.".format([chosen, rejected]))
            broken_data = True

        prompt = aligned_messages
        response = [
            {
                "role": tag_mapping.get(chosen.get(dataset_attr.role_tag, "gpt"), "assistant"),
                "content": chosen[dataset_attr.content_tag]
            },
            {
                "role": tag_mapping.get(rejected.get(dataset_attr.role_tag, "gpt"), "assistant"),
                "content": rejected[dataset_attr.content_tag]
            },
        ]

    else:  # normal example
        prompt = aligned_messages[:-1]
        response = aligned_messages[-1:]

    if broken_data:
        logger.warning("Skipping this abnormal example.")
        return outputs

    outputs["prompt"] = prompt
    outputs["response"] = response
    outputs["system"].append(system)
    outputs["tools"].append(sample[dataset_attr.tools] if dataset_attr.tools else "")
    for add_key in dataset_attr.dataset_additional_keys:
        if add_key != "labels":
            outputs[add_key] = sample[add_key]

    return outputs


def convert_hunyuan_to_intermediate(
    sample: Dict[str, List[Any]], dataset_attr: "InstructionDatasetAttr"):

    outputs = {"prompt": [], "response": [], "system": [], "tools": []}

    tag_mapping = {
        dataset_attr.user_tag: Role.USER.value,
        dataset_attr.assistant_tag: Role.ASSISTANT.value,
        dataset_attr.observation_tag: Role.OBSERVATION.value,
        dataset_attr.function_tag: Role.FUNCTION.value,
        dataset_attr.system_tag: Role.SYSTEM.value,
    }

    # "human" and "observation" must appear in odd-numbered positions
    # "gpt" and "function" must appear in even-numbered positions.
    sys_tags = (dataset_attr.system_tag)
    odd_tags = (dataset_attr.user_tag, dataset_attr.observation_tag)
    even_tags = (dataset_attr.assistant_tag, dataset_attr.function_tag)
    accept_tags = (sys_tags, odd_tags, even_tags)

    messages = sample[dataset_attr.messages]

    if len(messages) == 0:
        return outputs

    aligned_messages = []
    broken_data = False
    for turn_idx, message in enumerate(messages):
        if message[dataset_attr.role_tag] not in accept_tags[turn_idx % 3]:
            logger.warning("Invalid role tag in {}.".format(messages))
            broken_data = True

        content_value = message.get(dataset_attr.content_tag)

        if content_value is not None:
            aligned_messages.append(
                {"role": tag_mapping.get(message.get(dataset_attr.role_tag), "unknown"), "content": content_value}
            )
        else:
            logger.warning(f"Missing content tag in message at turn {turn_idx}: {message}")

    is_message_count_divisible_by_3 = len(aligned_messages) % 3 == 0
    if (not dataset_attr.ranking and not is_message_count_divisible_by_3) or (
        dataset_attr.ranking and is_message_count_divisible_by_3
    ):
        logger.warning("Invalid message count in {}.".format(messages))
        broken_data = True

    elif (
            dataset_attr.ranking
            and isinstance(sample[dataset_attr.chosen], dict)
            and isinstance(sample[dataset_attr.rejected], dict)
    ):
        chosen = sample[dataset_attr.chosen]
        rejected = sample[dataset_attr.rejected]
        if (
                chosen[dataset_attr.role_tag] not in accept_tags[-1]
                or rejected[dataset_attr.role_tag] not in accept_tags[-1]
        ):
            logger.warning("Invalid role tag in {}.".format([chosen, rejected]))
            broken_data = True

        prompt = aligned_messages
        response = [
            {
                "role": tag_mapping.get(chosen.get(dataset_attr.role_tag, "gpt"), "assistant"),
                "content": chosen[dataset_attr.content_tag]
            },
            {
                "role": tag_mapping.get(rejected.get(dataset_attr.role_tag, "gpt"), "assistant"),
                "content": rejected[dataset_attr.content_tag]
            },
        ]

    else:  # normal example
        prompt = aligned_messages[:-1]
        response = aligned_messages[-1:]

    if broken_data:
        logger.warning("Skipping this abnormal example.")
        return outputs

    outputs["prompt"] = prompt
    outputs["response"] = response
    outputs["system"] = ""
    outputs["tools"].append(sample[dataset_attr.tools] if dataset_attr.tools else "")
    return outputs


def convert_openai_to_intermediate(
    sample: Dict[str, List[Any]], dataset_attr: "InstructionDatasetAttr"
):
    """
    Convert OpenAI-format chat data (with structured tool_calls) into the
    framework's intermediate format.

    OpenAI format example:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "",
             "reasoning_content": "...",
             "tool_calls": [
                {"id": "...", "type": "function",
                 "function": {"name": "...", "arguments": "..."}}
             ]},
            {"role": "tool", "tool_call_id": "...", "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "tools": "[{...}]"
    }
    """
    outputs = {"prompt": [], "response": [], "system": [], "tools": []}

    # OpenAI uses standard role values directly. role_tag / content_tag etc.
    # come from dataset_attr (defaults to sharegpt's {from, value}, but users
    # can override via --map-keys for OpenAI's {role, content}).
    tag_mapping = {
        dataset_attr.user_tag: Role.USER.value,
        dataset_attr.assistant_tag: Role.ASSISTANT.value,
        dataset_attr.observation_tag: Role.OBSERVATION.value,
        dataset_attr.function_tag: Role.FUNCTION.value,
        dataset_attr.system_tag: Role.SYSTEM.value,
    }

    messages = sample[dataset_attr.messages]

    # System message extraction: prefer the leading system message if present,
    # otherwise fall back to the top-level `system` column (matches sharegpt
    # converter behaviour).
    if (
        dataset_attr.system_tag
        and len(messages) != 0
        and messages[0][dataset_attr.role_tag] == dataset_attr.system_tag
    ):
        system = messages[0][dataset_attr.content_tag]
        messages = messages[1:]
    else:
        system = sample[dataset_attr.system] if dataset_attr.system else ""

    if len(messages) == 0:
        return outputs

    aligned_messages = []
    for turn_idx, message in enumerate(messages):
        role_value = message.get(dataset_attr.role_tag)
        mapped_role = tag_mapping.get(role_value)
        if mapped_role is None:
            logger.warning(f"Unknown role tag {role_value!r} at turn {turn_idx}; skipping this example.")
            return outputs

        # Build the aligned message preserving all structured fields.
        # content can be None for assistant messages that only carry tool_calls;
        # normalize to "" so downstream templates don't have to special-case None.
        content = message.get(dataset_attr.content_tag)
        new_msg = {"role": mapped_role, "content": content if content is not None else ""}

        # Pass through structured fields used by tool-calling-aware templates.
        # Only attach when present — keeps simple chat samples clean.
        for field in ("tool_calls", "tool_call_id", "reasoning_content"):
            if field in message and message[field] is not None:
                new_msg[field] = message[field]

        aligned_messages.append(new_msg)

    # Lightweight validation: catch obviously broken samples but don't enforce
    # strict odd/even alternation. Tool-calling sequences legitimately violate
    # alternation (e.g. parallel tool calls produce consecutive tool messages).
    if aligned_messages[0]["role"] not in (Role.USER.value, Role.SYSTEM.value):
        logger.warning(
            f"First message must be user or system, got {aligned_messages[0]['role']!r}. "
            f"Skipping this abnormal example."
        )
        return outputs

    # SFT split: last message is response, the rest is prompt.
    if aligned_messages[-1]["role"] != Role.ASSISTANT.value:
        logger.warning(
            f"Last message must be assistant for SFT, got {aligned_messages[-1]['role']!r}. "
            f"Skipping this abnormal example."
        )
        return outputs

    prompt = aligned_messages[:-1]
    response = aligned_messages[-1:]

    # tools field: if it's already a JSON string, pass through; if it's a list
    # or dict (some datasets store it pre-parsed), serialize back to a string
    # since downstream template expects a string.
    tools = sample[dataset_attr.tools] if dataset_attr.tools else ""
    if isinstance(tools, (list, dict)):
        tools = json.dumps(tools, ensure_ascii=False)

    outputs["prompt"] = prompt
    outputs["response"] = response
    outputs["system"].append(system or "")
    outputs["tools"].append(tools)

    # Pass through additional dataset keys, matching sharegpt converter.
    for add_key in dataset_attr.dataset_additional_keys:
        if add_key != "labels":
            outputs[add_key] = sample[add_key]

    return outputs


def align_dataset(dataset, dataset_attr, data_args):
    """
    Aligned dataset:
        prompt: [{"role": "user", "content": "..."}] * (2T - 1)
        response: [{"role": "assistant", "content": "..."}]
        system: "..."
        tools: "...",
        images: []

    after doing convert_func, the features will be:
        features = Features.from_dict(
            {
                "prompt": [
                    {"role": {"dtype": "string", "_type": "Value"}, "content": {"dtype": "string", "_type": "Value"}}
                ],
                "response": [
                    {"role": {"dtype": "string", "_type": "Value"}, "content": {"dtype": "string", "_type": "Value"}}
                ],
                "system": [{"dtype": "string", "_type": "Value"}],
                "tools": [{"dtype": "string", "_type": "Value"}],
            }
        )
    """
    if dataset_attr.formatting == "alpaca":
        convert_func = partial(convert_alpaca_to_intermediate, dataset_attr=dataset_attr)
    elif data_args.handler_name == "HunyuanInstructionHandler":
        convert_func = partial(convert_hunyuan_to_intermediate, dataset_attr=dataset_attr)
    elif "OpenAI" in data_args.handler_name:
        convert_func = partial(convert_openai_to_intermediate, dataset_attr=dataset_attr)
    else:
        convert_func = partial(convert_sharegpt_to_intermediate, dataset_attr=dataset_attr)
    
    column_names = [k for k in next(iter(dataset)) if k not in dataset_attr.dataset_additional_keys]

    kwargs = dict(
        num_proc=data_args.workers,
        load_from_cache_file=(not data_args.overwrite_cache),
        desc="Converting format of dataset",
    )

    dataset = dataset.map(
        convert_func,
        remove_columns=column_names,
        **kwargs,
    )

    dataset = dataset.filter(lambda x: len(x["prompt"]) != 0 and len(x["response"]) != 0)
    return dataset


def merge_dataset(all_datasets, data_args):
    """Merging multiple Datasets by mix_strategy"""
    if len(all_datasets) == 1:
        return all_datasets[0]
    elif data_args.mix_strategy == "concat":
        if data_args.streaming:
            logger.warning("The samples between different datasets will not be mixed in streaming mode.")
        return concatenate_datasets(all_datasets)
    elif data_args.mix_strategy.startswith("interleave"):
        if not data_args.streaming:
            logger.warning("We recommend using `mix_strategy=concat` in non-streaming mode.")
        return interleave_datasets(
            datasets=all_datasets,
            probabilities=data_args.interleave_probs,
            seed=data_args.seed,
            stopping_strategy="first_exhausted" if data_args.mix_strategy.endswith("under") else "all_exhausted",
        )
    else:
        raise ValueError("Unknown mixing strategy.")


def load_single_dataset(dataset_attr, data_args):
    """loading single dataset by script/local file"""
    logger.info("Loading dataset {}...".format(dataset_attr))
    data_path, data_name, data_dir, data_files = None, None, None, None

    if dataset_attr.load_from == "script":
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "file":
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
                if data_path is None:
                    data_path = FILEEXT2TYPE.get(file_name.split(".")[-1], None)
                elif data_path != FILEEXT2TYPE.get(file_name.split(".")[-1], None):
                    raise ValueError("File types should be identical.")
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
            data_path = FILEEXT2TYPE.get(local_path.split(".")[-1], None)
        else:
            raise ValueError("File not found.")

        if data_path is None:
            raise ValueError("File extension must be txt, csv, arrow, parquet, json or jsonl.")
    else:
        raise NotImplementedError

    kwargs = {}
    split_flag = "train"
    dataset = load_dataset(
        path=data_path,
        name=data_name,
        data_dir=data_dir,
        split=split_flag,
        data_files=data_files,
        cache_dir=data_args.cache_dir,
        streaming=(data_args.streaming and (dataset_attr.load_from != "file")),
        **kwargs,
    )

    if data_args.max_samples is not None:  # truncate dataset
        num_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(num_samples))

    return align_dataset(dataset, dataset_attr, data_args)


def search_for_fit(numbers: Sequence[int], capacity: int) -> int:
    r"""
    Finds the index of largest number that fits into the knapsack with the given capacity.
    """
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


def greedy_knapsack(numbers: List[int], capacity: int) -> List[List[int]]:
    r"""
    An efficient greedy algorithm with binary search for the knapsack problem.
    """
    numbers.sort()  # sort numbers in ascending order for binary search
    knapsacks = []

    while numbers:
        current_knapsack = []
        remaining_capacity = capacity

        while True:
            index = search_for_fit(numbers, remaining_capacity)
            if index == -1:
                break  # no more numbers fit in this knapsack

            remaining_capacity -= numbers[index]  # update the remaining capacity
            current_knapsack.append(numbers.pop(index))  # add the number to knapsack

        knapsacks.append(current_knapsack)

    return knapsacks