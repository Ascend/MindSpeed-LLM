# Copyright 2024 Huawei Technologies Co., Ltd.

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

import abc
import torch
from megatron.training import get_args
from megatron.core import mpu
from megatron.core.pipeline_parallel.schedules import get_attr_wrapped_model
from mindspeed_llm.tasks.posttrain.utils import get_attr_from_wrapped_model


class HyperModelABC(abc.ABC):
    """
    The hyper model wraps multiple models required in reinforcement learning into a single model,
    maintaining the original distributed perspective unchanged.

    In different reinforcement learning scenarios, different numbers and states of models are required,
    and the following methods need to be implemented as needed.
    For example, in DPO, one trainable model and one frozen reference model are needed,
    while in PPO, three trainable models and one inference model for reward are required.
    """
    def __init__(self, *args, **kwargs):
        """
        You need to initialize each of the models and states.
        """

    def __call__(self, input_ids, position_ids, attention_mask):
        """
        Implementing the forward process of multiple models requires concatenating and chunking the inputs of
        multiple models along the batch dimension, paying attention to the perspective handling under PP parallelism.
        """

    def set_input_tensor(self):
        """
        In pipeline parallelism, you need to pass the output of the previous stage as input to the next stage,
        which requires custom implementation based on the scenario.

        For example, in DPO, we use the get_attr_from_wrapped_model method to get the original training model's
        set_input_tensor input [s, 4 * b, h] (which has already been replaced with the concatenated inputs of the
        training model and reference model, i.e., chosen and rejected of the training model and chosen and rejected of
        the reference model), and then separate them into the required inputs for each model
        and perform the actual set_input_tensor operation. In scenarios where pipeline parallelism is not used,
        such as tensor parallelism, the result of get_attr_from_wrapped_model is None,
        and this logic does not need to be executed.
        """


class DPOModel(HyperModelABC):
    """
    The hyper model wraps multiple models required in reinforcement learning into a single model,
    maintaining the original distributed perspective unchanged.
    """
    def __init__(self, train_model, refer_model):
        super().__init__()
        self.args = get_args()
        self.train_model = train_model
        self.refer_model = refer_model
        self._set_model_chunk_id()

        self.ori_micro_batch_size = self.args.micro_batch_size
        self.new_micro_batch_size = self.args.actual_micro_batch_size // 2

        self.input_tensor = None
        
    def _set_model_chunk_id(self):
        if self.args.virtual_pipeline_model_parallel_size is not None:
            self.model_chunk_id = mpu.get_virtual_pipeline_model_parallel_rank()
        else:
            self.model_chunk_id = 0

    def __call__(self, input_ids, position_ids, attention_mask):
        self._set_model_chunk_id()
        self.set_input_tensor()
        self.args.micro_batch_size = self.new_micro_batch_size

        if self.input_tensor is None:
            train_input_ids, refer_input_ids = (None, None) if input_ids is None else torch.chunk(input_ids, 2, dim=0)
            train_position_ids, refer_position_ids = (None, None) if position_ids is None else torch.chunk(
                position_ids, 2, dim=0)
            if isinstance(attention_mask, list):
                masks = [x.chunk(2, dim=0) for x in attention_mask]
                train_attention_mask, refer_attention_mask = list(zip(*masks))
                train_attention_mask, refer_attention_mask = list(train_attention_mask), list(refer_attention_mask)
            else:
                train_attention_mask, refer_attention_mask = (None, None) if attention_mask is None else torch.chunk(
                    attention_mask, 2, dim=0)
            refer_input_ids = refer_input_ids.detach()
            refer_position_ids = None if refer_position_ids is None else refer_position_ids.detach()
            if refer_attention_mask is not None:
                if isinstance(refer_attention_mask, list):
                    refer_attention_mask = [x.detach() for x in refer_attention_mask]
                else:
                    refer_attention_mask = refer_attention_mask.detach()
        else:
            train_input_ids, refer_input_ids = (None, None) if input_ids is None else torch.chunk(input_ids, 2, dim=0)
            train_position_ids, refer_position_ids = (None, None) if position_ids is None else torch.chunk(position_ids, 2, dim=0)
            if isinstance(attention_mask, list):
                masks = [x.chunk(2, dim=0) for x in attention_mask]
                train_attention_mask, refer_attention_mask = list(zip(*masks))
                train_attention_mask, refer_attention_mask = list(train_attention_mask), list(refer_attention_mask)
            else:
                train_attention_mask, refer_attention_mask = (None, None) if attention_mask is None else torch.chunk(
                    attention_mask, 2, dim=0)

        with torch.no_grad():
            refer_output = self.refer_model[self.model_chunk_id](refer_input_ids, refer_position_ids, refer_attention_mask)
        
        policy_output = self.train_model[self.model_chunk_id](train_input_ids, train_position_ids, train_attention_mask)

        if mpu.is_pipeline_last_stage():
            output_tensor = torch.cat((policy_output, refer_output), dim=0)
        else:
            output_tensor = torch.cat((policy_output, refer_output), dim=1)

        self.args.micro_batch_size = self.ori_micro_batch_size

        return output_tensor

    def set_input_tensor(self) -> None:
        """Sets input tensor to the hyper model.

        See megatron.model.transformer.set_input_tensor()
        """
        input_tensor = get_attr_from_wrapped_model(self.train_model[self.model_chunk_id], "input_tensor")

        if input_tensor[0] is not None:
            self.input_tensor = torch.chunk(input_tensor[0], 2, dim=1)

            set_train_input_tensor = get_attr_wrapped_model(self.train_model[self.model_chunk_id], "set_input_tensor")
            set_refer_input_tensor = get_attr_wrapped_model(self.refer_model[self.model_chunk_id], "set_input_tensor")
            set_train_input_tensor(self.input_tensor[0])
            set_refer_input_tensor(self.input_tensor[1])

    def set_is_first_microbatch(self) -> None:
        """Sets the is_first_microbatch flag if it exists. When this flag is set,
        TE modules will update their fp8 parameter cache.
        """
        self.train_model[self.model_chunk_id].set_is_first_microbatch()
        self.refer_model[self.model_chunk_id].set_is_first_microbatch()

    def zero_grad_buffer(self) -> None:
        self.train_model[self.model_chunk_id].zero_grad_buffer()

    def finish_grad_sync(self) -> None:
        self.train_model[self.model_chunk_id].finish_grad_sync()



