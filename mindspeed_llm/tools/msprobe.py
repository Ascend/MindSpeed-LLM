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

import logging
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)
_DEFAULT_CONFIG_PATH = Path(__file__).with_name("msprobe_config.json")


class MsProbeManager:
    """Manage optional msProbe precision-data collection for training steps."""

    def __init__(self, enabled: bool = False, config_path: Optional[str] = None):
        self.debugger = None
        self._started = False

        if not enabled:
            return
        config_file = Path(config_path).expanduser() if config_path else _DEFAULT_CONFIG_PATH
        if not config_file.is_file():
            raise FileNotFoundError(f"msProbe config file does not exist: {config_file}")

        try:
            from msprobe.pytorch import PrecisionDebugger
        except ImportError as exc:
            raise ImportError(
                "msProbe is enabled but `mindstudio-probe` is not installed. "
                "Install a version compatible with the current PyTorch environment."
            ) from exc

        self.debugger = PrecisionDebugger(config_path=str(config_file))
        logger.info("msProbe initialized with config: %s", config_file)

    @property
    def enabled(self) -> bool:
        return self.debugger is not None

    def set_init_step(self, step: int):
        if self.debugger is not None:
            self.debugger.set_init_step(step)

    def start_step(self, model: Any):
        if self.debugger is None:
            return
        if self._started:
            raise RuntimeError("msProbe collection has already started for the current step.")

        self.debugger.start(model=model)
        self._started = True

    def end_step(self):
        if self.debugger is None or not self._started:
            return

        try:
            self.debugger.stop()
            self.debugger.step()
        finally:
            self._started = False
