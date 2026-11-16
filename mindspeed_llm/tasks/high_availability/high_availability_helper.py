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
import warnings
from functools import wraps


def get_env_args(args):
    env = os.getenv('HIGH_AVAILABILITY', '')
    if not env:
        return args
    for strategy in env.split(','):
        if strategy.lower() in ('dump', 'recover', 'retry', 'elastic-training'):
            if not getattr(args, 'enable_high_availability', False):
                warnings.warn(
                    "HIGH_AVAILABILITY environment variables enabled and args.enable_high_availability inactive"
                )
            args.enable_high_availability = True
        if strategy.lower() == 'recover':
            args.enable_worker_reboot = True
        if strategy.lower() == 'retry':
            args.enable_hbmfault_repair = True
        if strategy.lower() == 'elastic-training':
            args.enable_elastic_training = True
    return args


def skip_reuse_register_patches(fn, argument):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not argument.enable_high_availability:
            fn(self, *args, **kwargs)
    return wrapper


def check_mindio_acp_available():
    try:
        import mindio_acp
        return True
    except Exception:
        return False
