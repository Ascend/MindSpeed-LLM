import sys
import shutil
import argparse
from logging import getLogger
from pathlib import Path
from multiprocessing import Lock

import torch
from torch.utils.cpp_extension import _get_build_directory
from torch_npu.contrib import transfer_to_npu
from mindspeed.features_manager.features_manager import MindSpeedFeaturesManager

LOG = getLogger(__name__)


def add_args(args, key, value):
    if key is not None:
        key = key[2:].replace('-', '_')
        if value is None:
            value = True
        elif len(value) == 1:
            value = value[0]
        setattr(args, key, value)


def parser_unknown_args(args, unknown):
    i = 0
    key = value = None
    while i < len(unknown):
        if unknown[i].startswith("--"):
            add_args(args, key, value)
            key = unknown[i]
            value = None
        else:
            if value is None:
                value = [unknown[i]]
            else:
                value.append(unknown[i])
        i += 1
    add_args(args, key, value)


class FeatureAdaptor:
    """
        A module manager supports adaptation registration, application and execution.
    """
    _args = None

    @classmethod
    def get_mindspeed_llm_args(cls):
        if cls._args is not None:
            return cls._args

        from mindspeed_llm.training.arguments import process_args_v2
        from mindspeed_llm.tasks.high_availability.high_availability_helper import get_env_args
        parser = argparse.ArgumentParser(description='MindSpeed-LLM Arguments', allow_abbrev=False)
        _args, unknown = process_args_v2(parser).parse_known_args()
        get_env_args(_args)
        parser_unknown_args(_args, unknown)
        return _args
    
    @classmethod
    def delete_lock_file(cls):
        """Delete lock file in multiprocess for JIT build."""
        directory = Path(_get_build_directory("", True))
        if not directory.exists():
            return
        with Lock():
            files = [item for item in directory.iterdir() if item.is_file() and item.name.endswith("lock")]
            if files:
                LOG.info("Process (PID:%s is deleting Lock directory", os.getpid())
                shutil.rmtree(directory)
   
    @classmethod
    def execute(cls):
        """
        Execute adaptations.
        """
        
        args = FeatureAdaptor.get_mindspeed_llm_args()
        FeatureAdaptor.delete_lock_file()
        
        # apply mindspeed base patches
        MindSpeedFeaturesManager.apply_features_pre_patches(args)
        # apply megatron patches
        MindSpeedFeaturesManager.apply_features_patches(args)
        
        # accelerate package will check TE on sys.modules, so we need remove this patch
        if 'transformer_engine' in sys.modules:
            del sys.modules["transformer_engine"]
    
FeatureAdaptor.execute()