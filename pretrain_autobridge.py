# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pretrain an AutoBridge-provided model with the MindSpeed-LLM loop."""

import os
import sys


if os.environ.get("MINDSPEED_LLM_VERSION") != "018":
    raise RuntimeError("pretrain_autobridge.py requires: export MINDSPEED_LLM_VERSION=018")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))

from megatron.core.enums import ModelType  # noqa: E402

try:
    from megatron.core.process_groups_config import ProcessGroupCollection as _ProcessGroupCollection  # noqa: E402,F401
except ImportError as exc:
    raise RuntimeError("The current Megatron-LM does not provide the ProcessGroupCollection API.") from exc

# Pylint resolves the legacy root package instead of the 018 tree added above.
from mindspeed_llm.training.autobridge import (  # noqa: E402  # pylint: disable=no-name-in-module
    add_autobridge_args,
    autobridge_model_provider,
    install_autobridge_feature,
)
from mindspeed_llm.training.autobridge_compat import (  # noqa: E402  # pylint: disable=no-name-in-module
    autobridge_training_scope,
    validate_autobridge_training_args,
)

# Load the Bridge Qwen classes against native MCore, before the normal LLM
# entrypoint installs its broader training patches.
install_autobridge_feature()

from pretrain_gpt import forward_step, get_embedding_ranks, train_valid_test_datasets_provider  # noqa: E402
from megatron.training.arguments import parse_and_validate_args  # noqa: E402  # pylint: disable=ungrouped-imports
from megatron.training.argument_utils import pretrain_cfg_container_from_args  # noqa: E402
from mindspeed_llm.training.training import pretrain  # noqa: E402  # pylint: disable=ungrouped-imports
from mindspeed_llm.training.utils import auto_coverage  # noqa: E402


@auto_coverage
def main():
    """Run the existing GPT data/loss loop with an AutoBridge model provider."""
    train_valid_test_datasets_provider.is_distributed = True
    # The current 0.18 training loop expects the caller to parse arguments and
    # build its configuration before distributed/model initialization.
    with autobridge_training_scope():
        args = parse_and_validate_args(
            extra_args_provider=add_autobridge_args,
            args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        )
        validate_autobridge_training_args(args)
        full_config = pretrain_cfg_container_from_args(args)
        # The 018 pretrain signature accepts get_embedding_ranks; the legacy one does not.
        pretrain(  # pylint: disable=unexpected-keyword-arg
            full_config,
            train_valid_test_datasets_provider,
            autobridge_model_provider,
            ModelType.encoder_or_decoder,
            forward_step,
            get_embedding_ranks=get_embedding_ranks,
        )


if __name__ == "__main__":
    main()
