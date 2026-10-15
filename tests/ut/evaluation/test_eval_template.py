# coding=utf-8
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
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
"""Unit tests for evaluation prompt templates."""

from pathlib import Path

import pandas as pd
import pytest

from mindspeed_llm.tasks.evaluation.eval_impl import template as eval_template
from tests.test_tools.utils import create_testconfig


def _example(question="Question?", answer="B"):
    return {
        "question": question,
        "A": "first",
        "B": "second",
        "C": "third",
        "D": "fourth",
        "answer": answer,
    }


class TestEvaluationTemplate:
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        ("name", "expected_answer"),
        test_config["test_builtin_templates_are_registered"],
    )
    def test_builtin_templates_are_registered(self, name, expected_answer):
        assert eval_template.get_eval_template(name).answer == expected_answer

    @pytest.mark.parametrize("name", test_config["test_get_eval_template_rejects_unknown_name"])
    def test_get_eval_template_rejects_unknown_name(self, name):
        with pytest.raises(ValueError, match=f"Template {name} does not exist"):
            eval_template.get_eval_template(name)

    @pytest.mark.parametrize(
        ("example", "expected_prompt", "expected_response"),
        test_config["test_parse_example_includes_only_available_choices"],
    )
    def test_parse_example_includes_only_available_choices(self, example, expected_prompt, expected_response):
        template = eval_template.get_eval_template("en")

        prompt, response = template._parse_example(example)

        assert prompt == expected_prompt
        assert response == expected_response

    @pytest.mark.parametrize(
        (
            "support_question",
            "support_answer",
            "target_question",
            "target_answer",
            "subject",
            "expected_roles",
        ),
        test_config["test_format_example_builds_few_shot_conversation"],
    )
    def test_format_example_builds_few_shot_conversation(
        self,
        support_question,
        support_answer,
        target_question,
        target_answer,
        subject,
        expected_roles,
    ):
        template = eval_template.get_eval_template("en")
        support_set = pd.DataFrame([_example(support_question, support_answer)])

        messages = template.format_example(_example(target_question, target_answer), support_set, subject)

        assert [message["role"] for message in messages] == expected_roles
        assert messages[0]["content"].startswith(
            "The following are multiple choice questions (with answers) about computer science.\n\nSupport question?"
        )
        assert messages[1] == {"role": "assistant", "content": "A"}
        assert messages[2]["content"].startswith("Target question?\nA. first")

    @pytest.mark.parametrize(
        ("question", "answer", "subject", "expected_content"),
        test_config["test_format_example_supports_zero_shot_prompt"],
    )
    def test_format_example_supports_zero_shot_prompt(self, question, answer, subject, expected_content):
        template = eval_template.get_eval_template("zh")

        messages = template.format_example(_example(question, answer), pd.DataFrame(), subject)

        assert messages == [{"role": "user", "content": expected_content}]

    def test_register_eval_template_makes_template_available(self, monkeypatch):
        name = "unit-test-template"
        monkeypatch.setitem(eval_template.eval_templates, name, None)

        eval_template._register_eval_template(name, "system {subject}", "{choice}:{content}", "answer", "prefix")

        registered = eval_template.get_eval_template(name)
        assert registered == eval_template.EvalTemplate(
            system="system {subject}",
            choice="{choice}:{content}",
            answer="answer",
            prefix="prefix",
        )
