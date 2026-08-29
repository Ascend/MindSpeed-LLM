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
from pathlib import Path
from dataclasses import dataclass
from mindspeed_llm.tasks.preprocess.templates import Role


cur_file_dir = Path(__file__).absolute().parent

GSM8K_TEMPLATE_DIR = os.path.join(cur_file_dir, "fewshot_template/gsm8k_3shot_template.json")
MMLU_TEMPLATE_DIR = os.path.join(cur_file_dir, "fewshot_template/mmlu_5shot_template.json")
CMMLU_TEMPLATE_DIR = os.path.join(cur_file_dir, "fewshot_template/cmmlu_5shot_template.json")
CEVAL_TEMPLATE_DIR = os.path.join(cur_file_dir, "fewshot_template/ceval_5shot_template.json")
AGIEVAL_TEMPLATE_DIR = os.path.join(cur_file_dir, "fewshot_template/AGI_fewshot.json")
BBH_TEMPLATE_DIR = os.path.join(cur_file_dir, "fewshot_template/bbh_template.json")
BBH_COT_TEMPLATE_DIR = os.path.join(cur_file_dir, "fewshot_template/bbh_cot_template.json")
CODE_TEST_LOG_DIR = os.path.join(cur_file_dir.parent, "codecheck_log")

CHOICES = ["A", "B", "C", "D"]


@dataclass
class EvalTemplate:
    system: str
    choice: str
    answer: str
    prefix: str

    def _parse_example(self, example):
        """
        input: a dict with keys {"question", "A", "B", "C", "D", "answer"}
        output: a tuple of (prompt, response)
        """
        candidates = [self.choice.format(choice=ch, content=example[ch]) for ch in CHOICES if ch in example]
        return "".join([example["question"]] + candidates + [self.answer]), example["answer"]

    def format_example(
        self, target_data, support_set, subject_name
    ):
        """
        Converts dataset examples to messages.
        """
        messages = []
        for _, row in support_set.iterrows():
            prompt, response = self._parse_example(row)
            messages.append({"role": Role.USER.value, "content": prompt})
            messages.append({"role": Role.ASSISTANT.value, "content": response})

        prompt, response = self._parse_example(target_data)
        messages.append({"role": Role.USER.value, "content": prompt})
        messages[0]["content"] = self.system.format(subject=subject_name) + messages[0]["content"]
        return messages


eval_templates = {}


def get_eval_template(name: str) -> "EvalTemplate":
    eval_template = eval_templates.get(name, None)
    if eval_template is None:
        raise ValueError(f"Template {name} does not exist.")
    return eval_template


def _register_eval_template(name: str, system: str, choice: str, answer: str, prefix: str) -> None:
    eval_templates[name] = EvalTemplate(system=system, choice=choice, answer=answer, prefix=prefix)


_register_eval_template(
    name="en",
    system="The following are multiple choice questions (with answers) about {subject}.\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: ",
    prefix=" ",
)


_register_eval_template(
    name="zh",
    system="以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n",
    choice="\n{choice}. {content}",
    answer="\n答案：",
    prefix=" ",
)