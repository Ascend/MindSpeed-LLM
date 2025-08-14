# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Sample Generate LLAMA"""
import os
import sys
import time
import logging
from typing import Union

from torch import distributed as dist
from transformers import AutoTokenizer
from mindspeed_llm import megatron_adaptor
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec, \
    get_gpt_layer_local_spec
from megatron.core.transformer.spec_utils import import_module
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args, print_rank_0
from megatron.legacy.model import GPTModel
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml

from mindspeed_llm.tasks.inference.module import GPTModelInfer, MegatronModuleForCausalLM
from mindspeed_llm.tasks.evaluation.eval_api.chat import Chat
from mindspeed_llm.tasks.evaluation.eval_impl.boolq_eval import BoolqEval
from mindspeed_llm.tasks.evaluation.eval_impl.gsm8k_eval import Gsm8kEval
from mindspeed_llm.tasks.evaluation.eval_impl.mmlu_eval import MmluEval
from mindspeed_llm.tasks.evaluation.eval_impl.mmlu_ppl import MmluEval_PPL
from mindspeed_llm.tasks.evaluation.eval_impl.ceval_exam import CEvalExam
from mindspeed_llm.tasks.evaluation.eval_impl.bbh_eval import BBHEval
from mindspeed_llm.tasks.evaluation.eval_impl.agi_eval import AGIEvalExam
from mindspeed_llm.tasks.evaluation.eval_impl.human_eval import HumanEval
from mindspeed_llm.tasks.evaluation.eval_impl.cmmlu_eval import CmmluEval
from mindspeed_llm.tasks.evaluation.eval_impl.needlebench_eval import NeedleBenchEval

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def model_provider(pre_process=True, post_process=True) -> Union[GPTModelInfer, GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModelInfer, GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_mcore_models:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)

        model = GPTModelInfer(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True if args.sequence_parallel else False,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor
        )
    else:
        if not args.context_parallel_size == 1:
            raise ValueError("Context parallelism is only supported with Megatron Core!")

        model = GPTModel(
            config,
            parallel_output=True if args.sequence_parallel else False,
            pre_process=pre_process,
            post_process=post_process
        )

    return model


def get_result(result, tokenizer):
    if result:
        final_results = []
        if isinstance(result[0], list):
            for idx, res in enumerate(result[0]):
                final_result = [res]
                if result[1][idx][0][tokenizer.encode("Yes")[-1]] >= result[1][idx][0][tokenizer.encode("No")[-1]]:
                    final_result.append('T')
                else:
                    final_result.append('F')
                final_results.append(final_result)
        else:
            final_result = [result[0]]
            if result[1][0][tokenizer.encode("Yes")[-1]] >= result[1][0][tokenizer.encode("No")[-1]]:
                final_result.append('T')
            else:
                final_result.append('F')
            final_results.append(final_result)
    else:
        final_results = None
    return final_results


class LLMChat(Chat):
    def __init__(self, llm_args, model, tokenizer):
        self.args = llm_args
        self.model = model
        self.tokenizer = tokenizer
        self.template = "{instruction}"

    def chat(self, instruction, history):
        instruction_temp = None
        if getattr(self.args, "task", False) and self.args.task[0] == 'needlebench':
            instruction_temp = [self.tokenizer.apply_chat_template([{"role": "user", "content": ins + '\n'}], add_generation_prompt=True, tokenize=False) for ins in instruction]
        elif self.args.prompt_type is None:
            instruction_temp = [self.template.format(instruction=ins) if (self.tokenizer.chat_template is None or self.args.no_chat_template) else self.tokenizer.apply_chat_template([{"role": "user", "content": ins}]) for ins in instruction]
        else:
            instruction_temp = instruction

        return_output_log_probs = False if (getattr(self.args, "task", False) and self.args.task[0] == 'needlebench') else True
        result = self.model.generate(
            instruction_temp,
            do_sample=False,
            max_new_tokens=self.args.max_new_tokens,
            stream=False,
            return_output_log_probs=return_output_log_probs,
            broadcast=self.args.broadcast
        )
        if getattr(self.args, "task", False) and self.args.task[0] == 'needlebench':
            return result, dist.get_rank()
        return get_result(result, self.tokenizer), dist.get_rank()

    def beam_search_chat(self, instruction, history):
        instruction_temp = None
        if self.args.prompt_type is None:
            instruction_temp = self.template.format(instruction=instruction) if (self.tokenizer.chat_template is None or self.args.no_chat_template) else self.tokenizer.apply_chat_template([{"role": "user", "content": instruction}])
        else:
            instruction_temp = instruction
        
        if "human_eval" in self.args.task and self.args.alternative_prompt:
            result = self.model.generate(
                instruction_temp,
                do_sample=False,
                max_new_tokens=self.args.max_new_tokens,
                stream=False
            )
        else:
            result = self.model.generate(
                instruction_temp,
                do_sample=False,
                max_new_tokens=self.args.max_new_tokens,
                stream=False,
                num_beams=4,
                top_k=50,
                top_p=0.95,
                length_penalty=0.7
            )
        return [result], dist.get_rank()


def mmlu(eval_args, agent):
    data_path = None
    answer = None 
    score_df = None
    for path in eval_args.task_data_path:
        data_path = path
    try:
        if data_path:
            mmlu_eval = MmluEval(test_dir=data_path, eval_args=eval_args)
            answer, score_df = mmlu_eval.eval(chat=agent)
            if dist.get_rank() == 0:
                logger.info('\n{}'.format(score_df))
    except Exception as e:
        logger.info(e)

    return answer, score_df


def cmmlu(eval_args, agent):
    data_path = None
    answer = None 
    score_df = None
    for path in eval_args.task_data_path:
        data_path = path
    try:
        if data_path:
            cmmlu_eval = CmmluEval(test_dir=data_path, eval_args=eval_args)
            answer, score_df = cmmlu_eval.eval(chat=agent)
            if dist.get_rank() == 0:
                logger.info('\n{}'.format(score_df))
    except Exception as e:
        logger.info(e)
    return answer, score_df


def needlebench(eval_args, agent):
    data_path = None
    for path in eval_args.task_data_path:
        data_path = path
    try:
        if data_path:
            needlebench_eval = NeedleBenchEval(test_dir=data_path, eval_args=eval_args)
            needlebench_eval.eval(chat=agent)
    except Exception as e:
        logger.info(e)

    return


def mmlu_ppl(eval_args, agent):
    data_path = None
    answer = None 
    score_df = None
    for path in eval_args.task_data_path:
        if 'mmlu' in path:
            data_path = path
    try:
        if data_path:
            mmlu_ppl_eval = MmluEval_PPL(test_dir=data_path, eval_args=eval_args)
            answer, score_df = mmlu_ppl_eval.eval(chat=agent)
            if dist.get_rank() == 0:
                logger.info('\n{}'.format(score_df))
    except Exception as e:
        logger.info(e)


def gsm8k(eval_args, agent):
    data_path = None
    answer = None 
    score_df = None
    for path in eval_args.task_data_path:
        data_path = path
    try:
        if data_path:
            gsm8k_eval = Gsm8kEval(test_dir=data_path, eval_args=eval_args)
            answer, score_df = gsm8k_eval.eval(chat=agent)
            if dist.get_rank() == 0:
                logger.info('\n{}'.format(score_df))
    except Exception as e:
        logger.info(e)

    return answer, score_df


def boolq(eval_args, agent):
    data_path = None
    answer = None 
    score_df = None

    for path in eval_args.task_data_path:
        data_path = path
    try:
        if data_path:
            boolq_eval = BoolqEval(test_dir=data_path, eval_args=eval_args)
            answer, score_df = boolq_eval.eval(chat=agent)
            if dist.get_rank() == 0:
                logger.info('\n{}'.format(score_df))
    except Exception as e:
        logger.info(e)

    return answer, score_df


def ceval(eval_args, agent):
    data_path = None
    answer = None 
    score_df = None

    for path in eval_args.task_data_path:
        data_path = path
    try:
        if data_path:
            ceval_exam = CEvalExam(test_dir=data_path, eval_args=eval_args)
            answer, score_df = ceval_exam.eval(chat=agent)
            if dist.get_rank() == 0:
                logger.info('\n{}'.format(score_df))
    except Exception as e:
        logger.info(e)

    return answer, score_df


def human_eval(eval_args, agent):
    data_path = None
    answer = None 
    score_df = None

    for path in eval_args.task_data_path:
        data_path = path
    try:
        if data_path:
            human_eval_exam = HumanEval(test_dir=data_path, eval_args=eval_args)
            answer, score_df = human_eval_exam.eval(chat=agent)
            if dist.get_rank() == 0:
                logger.info('\n{}'.format(score_df))
    except Exception as e:
        logger.info(e)

    return answer, score_df


def agi_eval(eval_args, agent):
    data_path = None
    answer = None 
    score_df = None

    for path in eval_args.task_data_path:
        data_path = path
    try:
        if data_path:
            agieval_exam = AGIEvalExam(test_dir=data_path, eval_args=eval_args)
            answer, score_df = agieval_exam.eval(chat=agent)
            if dist.get_rank() == 0:
                logger.info('\n{}'.format(score_df))
    except Exception as e:
        logger.info(e)

    return answer, score_df


def bbh_eval(eval_args, agent):
    data_path = None
    answer = None 
    score_df = None

    for path in eval_args.task_data_path:
        data_path = path
    try:
        if data_path:
            bbh = BBHEval(test_dir=data_path, eval_args=eval_args)
            answer, score_df = bbh.eval(chat=agent)
            if dist.get_rank() == 0:
                logger.info('\n{}'.format(score_df))
    except Exception as e:
        logger.info(e)

    return answer, score_df


def main():
    initialize_megatron(args_defaults={'no_load_rng': True,
                                       'no_load_optim': True})
    args = get_args()
    model = MegatronModuleForCausalLM.from_pretrained(
        model_provider=model_provider,
        pretrained_model_name_or_path=args.load
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, trust_remote_code=True, local_files_only=True)

    rank = dist.get_rank()
    if 'cmmlu' in args.task:
        a = time.time()
        cmmlu(args, LLMChat(args, model, tokenizer))
        if rank == 0:
            logger.info(f'CMMLU Running Time:, {time.time() - a}')
    if 'mmlu_ppl' in args.task:
        a = time.time()
        mmlu_ppl(args, LLMChat(args, model, tokenizer))
        if rank == 0:
            logger.info(f'MMLU_PPL Running Time:, {time.time() - a}')
    if 'mmlu' in args.task:
        a = time.time()
        mmlu(args, LLMChat(args, model, tokenizer))
        if rank == 0:
            logger.info(f'MMLU Running Time:, {time.time() - a}')
    if 'gsm8k' in args.task:
        a = time.time()
        gsm8k(args, LLMChat(args, model, tokenizer))
        if rank == 0:
            logger.info(f'GSM8k Running Time: {time.time() - a}')
    if 'boolq' in args.task:
        a = time.time()
        boolq(args, LLMChat(args, model, tokenizer))
        if rank == 0:
            logger.info(f'Boolq Running Time: {time.time() - a}')
    if 'ceval' in args.task:
        a = time.time()
        ceval(args, LLMChat(args, model, tokenizer))
        if rank == 0:
            logger.info(f'Ceval Running Time: {time.time() - a}')
    if 'bbh' in args.task:
        a = time.time()
        bbh_eval(args, LLMChat(args, model, tokenizer))
        if rank == 0:
            logger.info(f'bbh Running Time: {time.time() - a}')
    if 'agieval' in args.task:
        a = time.time()
        agi_eval(args, LLMChat(args, model, tokenizer))
        if rank == 0:
            logger.info(f'agi_eval Running Time: {time.time() - a}')
    if 'human_eval' in args.task:
        a = time.time()
        human_eval(args, LLMChat(args, model, tokenizer))
        if rank == 0:
            logger.info(f'Human_eval Running Time: {time.time() - a}')
    if 'needlebench' in args.task:
        a = time.time()
        needlebench(args, LLMChat(args, model, tokenizer))
        if rank == 0:
            logger.info(f'NeedleBench_eval Running Time: {time.time() - a}')



if __name__ == "__main__":
    main()

