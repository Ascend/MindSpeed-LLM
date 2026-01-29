# profile.py
import os
import json
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

import torch
import torch_npu.profiler as npu_profiler
from mindspeed_llm.fsdp2.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProfilerConfig:
    enabled: bool = False
    profile_step_start: int = 0          # inclusive
    profile_step_end: int = -1           # exclusive; -1 means until end
    profile_ranks: List[int] = None      # e.g., [0, 2, 4] or [-1] for all
    profile_level: str = "level0"        # 'level_none', 'level0', 'level1', 'level2'
    profile_export_type: str = "text"    # 'text' or 'db'
    profile_data_simplification: bool = False
    profile_with_cpu: bool = False
    profile_with_stack: bool = False
    profile_with_memory: bool = False
    profile_record_shapes: bool = False
    profile_save_path: str = "./profile"
    current_rank: int = 0

    def __post_init__(self):
        if self.profile_ranks is None:
            self.profile_ranks = [-1]

    def is_profiling_rank(self) -> bool:
        if not self.enabled:
            return False
        if -1 in self.profile_ranks:
            return True
        return self.current_rank in self.profile_ranks


class ProfilerManager:
    def __init__(self, config: ProfilerConfig):
        self.config = config
        self.profiler = None
        self._started = False

        if not config.is_profiling_rank():
            return

        Path(config.profile_save_path).mkdir(parents=True, exist_ok=True)

        # --- Level ---
        level_map = {
            "level_none": npu_profiler.ProfilerLevel.Level_none,
            "level0": npu_profiler.ProfilerLevel.Level0,
            "level1": npu_profiler.ProfilerLevel.Level1,
            "level2": npu_profiler.ProfilerLevel.Level2,
        }
        if config.profile_level not in level_map:
            raise ValueError(f"Invalid profile_level: {config.profile_level}")
        profiler_level = level_map[config.profile_level]

        # --- Export Type ---
        export_map = {
            "text": npu_profiler.ExportType.Text,
            "db": npu_profiler.ExportType.Db,
        }
        if config.profile_export_type not in export_map:
            raise ValueError(f"Invalid profile_export_type: {config.profile_export_type}")
        profile_export_type = export_map[config.profile_export_type]

        # --- Schedule (no resume, iteration = 0) ---
        if config.profile_step_end == -1:
            active = 1000000
        else:
            active = config.profile_step_end - config.profile_step_start
            if active <= 0:
                raise ValueError("profile_step_end must be > profile_step_start")

        skip_first = max(0, config.profile_step_start - 1)
        warmup = 0 if config.profile_step_start == 0 else 1

        # --- Activities ---
        activities = [npu_profiler.ProfilerActivity.NPU]
        if config.profile_with_cpu:
            activities.append(npu_profiler.ProfilerActivity.CPU)

        # --- Experimental Config ---
        experimental_config = npu_profiler._ExperimentalConfig(
            aic_metrics=npu_profiler.AiCMetrics.PipeUtilization,
            profiler_level=profiler_level,
            export_type=profile_export_type,
            data_simplification=config.profile_data_simplification,
        )

        # --- Create Profiler ---
        self.profiler = npu_profiler.profile(
            activities=activities,
            schedule=npu_profiler.schedule(
                wait=0, warmup=warmup, active=active, repeat=1, skip_first=skip_first
            ),
            on_trace_ready=npu_profiler.tensorboard_trace_handler(config.profile_save_path),
            record_shapes=config.profile_record_shapes,
            profile_memory=config.profile_with_memory,
            with_stack=config.profile_with_stack,
            experimental_config=experimental_config,
        )

        # --- Metadata ---
        try:
            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            self.profiler.add_metadata_json("distributed_args", json.dumps({
                "rank": config.current_rank,
                "world_size": world_size,
            }))
        except Exception as e:
            logger.warning(f"Failed to add meta {e}")

    def start(self):
        if self.profiler is not None and not self._started:
            self.profiler.start()
            self._started = True
            logger.info_rank0(f"[RANK {self.config.current_rank}] Profiling started.")

    def step(self):
        if self.profiler is not None:
            self.profiler.step()

    def stop(self):
        if self.profiler is not None:
            self.profiler.stop()
            logger.info_rank0(f"[RANK {self.config.current_rank}] Profiling stopped. Trace saved to {self.config.profile_save_path}")