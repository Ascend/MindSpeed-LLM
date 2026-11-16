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

import os
import re

import torch

retry_step_ = None


def handle_precision_error(err_str):
    match = re.search(r"load ckpt step=(\d+)", err_str)
    if match:
        global retry_step_
        retry_step_ = int(match.group(1))

    return "PRECISION ERROR"


def modify_ckpt_step(ckpt_path: str):
    global retry_step_
    if retry_step_ is None or ckpt_path is None:
        return

    directory = 'iter_{:07d}'.format(retry_step_)
    common_path = os.path.join(ckpt_path, directory)
    step_file = os.path.join(ckpt_path, "latest_checkpointed_iteration.txt")
    exist = os.path.exists(step_file) and os.path.exists(common_path)
    if not exist:
        return

    if torch.distributed.get_rank() == 0:
        with open(step_file, 'w') as f:
            f.write(str(retry_step_))

    torch.distributed.barrier()
    retry_step_ = None
