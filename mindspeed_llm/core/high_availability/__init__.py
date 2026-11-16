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

from .tft_replica_optimizer import TTPReplicaOptimizer
from .tft_replica_chainedoptimizer import TTPReplicaChainedOptimizer

from .tft_replica_optimizer_fp16 import TTPFP16ReplicaOptimizer
from .tft_replica_group import (ttp_get_replica_dp_num, ttp_get_dp_cp_replica_group, ttp_get_dp_cp_replica_group_gloo,
                                ttp_get_dp_ep_replica_group, ttp_get_dp_ep_replica_group_gloo,
                                ttp_initialize_replica_dp_group, tft_get_node_group)
from .tft_arf_group_repair import tft_is_arf_reboot_node
from .tft_train_initialize import tft_train, tft_init_controller_processor, tft_register_processor
from .tft_optimizer_data_repair import tft_set_losses_reduced
from .elastic_training_register import register_callbacks