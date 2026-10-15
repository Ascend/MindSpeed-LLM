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
"""Unit tests for evaluation prompt formatting and answer post-processing."""

from pathlib import Path

import pytest

from mindspeed_llm.tasks.evaluation.eval_utils import (
    agi_utils,
    bbh_utils,
    boolq_utils,
    ceval_utils,
    cmmlu_utils,
    gsm8k_utils,
    mmlu_utils,
    needlebench_utils,
)
from tests.test_tools.utils import create_testconfig


class _SupportSet:
    """Minimal dataframe-like object needed by the prompt formatters."""

    def __init__(self, *rows):
        self.rows = rows

    def iterrows(self):
        return enumerate(self.rows)


class TestEvaluationPostprocess:
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        ("item", "expected"),
        test_config["test_agi_default_instruction_handles_labels_and_free_form_answers"],
    )
    def test_agi_default_instruction_handles_labels_and_free_form_answers(self, item, expected):
        assert agi_utils.get_default_instruction(item) == tuple(expected)

    def test_agi_alternative_prompt_formats_chinese_multiple_choice(self):
        item = {
            "passage": "阅读材料。",
            "question": "请选择。",
            "options": ["A. 一", "B. 二", "C. 三"],
            "label": "B",
            "answer": None,
        }

        prompt, correct, choices = agi_utils.alternativate_prompt_instruction(item, "gaokao-chinese")

        assert prompt == (
            "以下是一道中国高考语文选择题，请选择正确的答案。\n阅读材料。请选择。\nA. 一\nB. 二\nC. 三\n答案是： "
        )
        assert correct == "B"
        assert choices == "ABC"

    def test_agi_alternative_prompt_formats_english_cloze_without_choices(self):
        item = {
            "passage": "",
            "question": "What is 6 x 7?",
            "options": [],
            "label": None,
            "answer": "$42$",
        }

        prompt, correct, choices = agi_utils.alternativate_prompt_instruction(item, "math")

        assert prompt == (
            "The following is a Math question. Please select the correct answer.\nWhat is 6 x 7?\nThe answer is "
        )
        assert correct == "42"
        assert choices is None

    @pytest.mark.parametrize(
        ("label", "expected_correct"),
        test_config["test_agi_alternative_prompt_handles_list_label_and_missing_answer"],
    )
    def test_agi_alternative_prompt_handles_list_label_and_missing_answer(self, label, expected_correct):
        item = {
            "passage": "",
            "question": "Select if possible.",
            "options": [],
            "label": label,
            "answer": None,
        }

        _, correct, choices = agi_utils.alternativate_prompt_instruction(item, "math")

        assert correct == expected_correct
        assert choices is None

    @pytest.mark.parametrize(
        ("length", "expected"),
        test_config["test_agi_generate_alphabet_string"],
    )
    def test_agi_generate_alphabet_string(self, length, expected):
        assert agi_utils.generate_alphabet_string(length) == expected

    def test_agi_selects_subject_specific_postprocessor(self):
        multi = agi_utils.get_pred_postprocess_func("gaokao-physics", "ABCD")
        single = agi_utils.get_pred_postprocess_func("sat-math", "ABCD")
        cloze = agi_utils.get_pred_postprocess_func("math", None)

        assert multi("answers AC") == "AC"
        assert single("The answer is C") == "C"
        assert cloze("The answer is therefore $x = 9$") == "9"

    def test_agi_rejects_unknown_subject_postprocessor(self):
        with pytest.raises(ValueError, match="Unknown subject_name: unsupported"):
            agi_utils.get_pred_postprocess_func("unsupported", "ABCD")

    @pytest.mark.parametrize(
        ("text", "expected"),
        test_config["test_agi_first_option_postprocess_matches_chinese_and_english"],
    )
    def test_agi_first_option_postprocess_matches_chinese_and_english(self, text, expected):
        assert agi_utils.first_option_postprocess(text, "ABCD", cushion=False) == expected

    def test_agi_first_option_postprocess_cushion_is_optional(self):
        text = "model-output=C:"

        assert agi_utils.first_option_postprocess(text, "ABCD", cushion=True) == "C"
        assert agi_utils.first_option_postprocess(text, "ABCD", cushion=False) == ""

    @pytest.mark.parametrize(
        ("text", "expected"),
        test_config["test_agi_multiple_choice_postprocess"],
    )
    def test_agi_multiple_choice_postprocess(self, text, expected):
        assert agi_utils.first_capital_postprocess_multi(text) == expected

    @pytest.mark.parametrize(
        ("raw_answer", "expected"),
        test_config["test_agi_parse_math_answer_handles_common_answer_notations"],
    )
    def test_agi_parse_math_answer_handles_common_answer_notations(self, raw_answer, expected):
        assert agi_utils.parse_math_answer(raw_answer) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        test_config["test_agi_remove_few_shot_prefix_uses_latest_answer"],
    )
    def test_agi_remove_few_shot_prefix_uses_latest_answer(self, text, expected):
        assert agi_utils.remove_few_shot_prefix(text) == expected

    def test_mmlu_parse_and_format_examples(self):
        support = _SupportSet(
            {
                "question": "Support question?",
                "A": "one",
                "B": "two",
                "C": "three",
                "D": "four",
                "answer": "B",
            }
        )
        target = {
            "question": "Target question?",
            "A": "red",
            "B": "blue",
            "C": "green",
            "D": "black",
            "answer": "C",
        }

        parsed, answer = mmlu_utils._parse_example(target)
        formatted = mmlu_utils._format_example(target, support, "computer_science")

        assert parsed == ("Target question?\nA. red\nB. blue\nC. green\nD. black\nAnswer: ")
        assert answer == "C"
        assert "about computer science" in formatted
        assert "Support question?" in formatted
        assert "\nAnswer: \nB\n\n" in formatted
        assert formatted.endswith(parsed)
        assert not formatted.endswith("C")

    def test_mmlu_parse_example_omits_missing_choices(self):
        parsed, answer = mmlu_utils._parse_example({"question": "Binary?", "A": "yes", "B": "no", "answer": "A"})

        assert parsed == "Binary?\nA. yes\nB. no\nAnswer: "
        assert answer == "A"

    def test_mmlu_postprocess_unwraps_nested_generation_and_uses_gold_fast_path(self):
        generation = [["C is the first-line answer\nignored"]]

        assert mmlu_utils.postprocess(generation, "ABCD", "C", False) == "C"

    @pytest.mark.parametrize(
        ("text", "origin_postprocess", "expected"),
        test_config["test_mmlu_postprocess_fallback_patterns"],
    )
    def test_mmlu_postprocess_fallback_patterns(self, text, origin_postprocess, expected):
        assert mmlu_utils.postprocess(text, "ABCD", "A", origin_postprocess, cushion=False) == expected

    def test_mmlu_postprocess_cushion_is_optional(self):
        text = "model-output=C:"

        assert mmlu_utils.postprocess(text, "ABCD", "A", True, cushion=True) == "C"
        assert mmlu_utils.postprocess(text, "ABCD", "A", True, cushion=False) == ""

    def test_mmlu_formats_perplexity_prompt_with_support_answer_only(self):
        support = _SupportSet({"question": "Support?", "A": 1, "B": 2, "answer": "A"})
        target = {"question": "Target?", "A": 3, "B": 4, "answer": "B"}

        parsed, answer = mmlu_utils.parse_ppl_example(target)
        prompt = mmlu_utils.format_ppl_prompt(target, support, "abstract_algebra")

        assert parsed == "Target?\nA. 3\nB. 4\nAnswer: "
        assert answer == "B"
        assert prompt.startswith("The following are multiple choice questions (with answers) about  abstract algebra.")
        assert "Support?\nA. 1\nB. 2\nAnswer: A\n\n" in prompt
        assert prompt.endswith(parsed)

    def test_cmmlu_loads_repository_mapping_and_parses_example(self):
        example = {
            "question": "哪一个正确？",
            "A": "甲",
            "B": "乙",
            "C": "丙",
            "D": "丁",
            "answer": "D",
        }

        prompt, answer = cmmlu_utils._parse_example(example)

        assert cmmlu_utils.cmmlu_subject_mapping["agronomy"] == "农学"
        assert prompt == "哪一个正确？\nA. 甲\nB. 乙\nC. 丙\nD. 丁\n答案是: "
        assert answer == "D"

    @pytest.mark.parametrize(
        ("language", "expected_prefix"),
        test_config["test_cmmlu_format_example_supports_chinese_and_english"],
    )
    def test_cmmlu_format_example_supports_chinese_and_english(self, language, expected_prefix):
        support = _SupportSet({"question": "Support?", "A": "1", "B": "2", "answer": "A"})
        target = {"question": "Target?", "A": "3", "B": "4", "answer": "B"}

        prompt = cmmlu_utils.cmmlu_format_example(target, support, "农学" if language == "zh" else "agronomy", language)

        assert prompt.startswith(expected_prefix)
        assert "Support?\nA. 1\nB. 2\n答案是: A\n" in prompt
        assert prompt.endswith("Target?\nA. 3\nB. 4\n答案是: ")
        assert not prompt.endswith("B")

    def test_cmmlu_postprocess_uses_configured_and_cushion_patterns(self):
        assert cmmlu_utils.first_option_postprocess("因此答案是： C", "ABCD", False) == "C"
        assert cmmlu_utils.first_option_postprocess("model-output=D", "ABCD", True) == "D"
        assert cmmlu_utils.first_option_postprocess("model-output=D", "ABCD", False) == ""
        assert cmmlu_utils.first_option_postprocess("no choice", "ABCD", False) == ""

    def test_ceval_formats_few_shot_and_target_templates(self):
        support_row = {
            "question": "支持题？",
            "A": "甲",
            "B": "乙",
            "C": "丙",
            "D": "丁",
            "answer": "A",
        }
        target = {
            "question": "目标题？",
            "A": "一",
            "B": "二",
            "C": "三",
            "D": "四",
            "answer": "C",
        }

        support_prompt = ceval_utils.parse_templates(support_row, "计算机网络")
        prompt = ceval_utils.format_ceval_templates(target, _SupportSet(support_row), "computer_network")

        assert support_prompt.endswith("答案: \nA\n")
        assert prompt.startswith(support_prompt)
        assert "以下是中国关于计算机网络考试的单项选择题" in prompt
        assert prompt.endswith("目标题？\nA. 一\nB. 二\nC. 三\nD. 四\n答案: ")
        assert not prompt.endswith("C")

    @pytest.mark.parametrize(
        ("postprocessor_name", "text", "container_type", "expected"),
        test_config["test_ceval_and_boolq_first_capital_postprocessors"],
    )
    def test_ceval_and_boolq_first_capital_postprocessors(self, postprocessor_name, text, container_type, expected):
        postprocessor = {
            "ceval": ceval_utils.first_capital_postprocess,
            "boolq": boolq_utils.first_capital_postprocess,
        }[postprocessor_name]
        if container_type == "tuple":
            text = tuple(text)

        assert postprocessor(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        test_config["test_bbh_multiple_choice_postprocess"],
    )
    def test_bbh_multiple_choice_postprocess(self, text, expected):
        assert bbh_utils.bbh_mcq_postprocess(text) == expected

    @pytest.mark.parametrize(
        ("text", "subject", "expected"),
        test_config["test_bbh_freeform_postprocess"],
    )
    def test_bbh_freeform_postprocess(self, text, subject, expected):
        assert bbh_utils.bbh_freeform_postprocess(text, subject) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        test_config["test_gsm8k_postprocess_returns_last_number_before_next_question"],
    )
    def test_gsm8k_postprocess_returns_last_number_before_next_question(self, text, expected):
        assert gsm8k_utils.gsm8k_postprocess(text) == expected

    @pytest.mark.parametrize(
        ("prediction", "reference", "expected"),
        test_config["test_needlebench_trim_prediction_uses_reference_length_and_terminator"],
    )
    def test_needlebench_trim_prediction_uses_reference_length_and_terminator(self, prediction, reference, expected):
        assert needlebench_utils.trim_prediction(prediction, reference) == expected

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        test_config["test_needlebench_levenshtein_distance"],
    )
    def test_needlebench_levenshtein_distance(self, left, right, expected):
        assert needlebench_utils.levenshtein_distance(left, right) == expected
