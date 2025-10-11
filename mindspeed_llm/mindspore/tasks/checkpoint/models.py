# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import abc
import os
import sys
import re
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from mindspeed_llm.tasks.checkpoint.models import ModelBase


def register_functions(self):
    self.get_module_mapping()

    def _get_obj(self, value, **kwargs):
        pattern = r'(\w+)(?:\[(\w+)\])?'
        matches = re.findall(pattern, value)
        self.update_kwargs_idx(**kwargs)
        obj = self.get_model_item(**kwargs)
        for attr, attr_ident in matches:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                return None
            if attr_ident:
                if attr_ident in self.kwargs_idx:
                    attr_idx = self.kwargs_idx[attr_ident]
                    obj = obj[attr_idx]
                else:
                    raise AssertionError(f"check {self.__class__.__name__}.module_mapping **{attr_ident}**.")
        return obj

    def _get_dst_obj(self, value, **kwargs):
        if kwargs.get("layer_idx") is None:
            kwargs["layer_idx"] = kwargs.get("dst_layer_idx")

        return _get_obj(self, value, **kwargs)

    def _get_src_obj(self, value, **kwargs):
        if kwargs.get("layer_idx") is None:
            kwargs["layer_idx"] = kwargs.get("src_layer_idx")

        return _get_obj(self, value, **kwargs)

    def _func_generator_get_module(value):
        def func(self, **kwargs):
            return _get_src_obj(self, value, **kwargs)
        return func

    def _func_generator_get_weight(value):
        def func(self, **kwargs):
            return _get_src_obj(self, value, **kwargs).weight.data
        return func

    def _func_generator_get_bias(value):
        def func(self, **kwargs):
            return _get_src_obj(self, value, **kwargs).bias.data
        return func

    def _func_generator_set_weight(value):
        def func(self, **kwargs):
            set_tensor = _get_dst_obj(self, value, **kwargs)
            data = kwargs.get('data')
            if data.dtype != set_tensor.weight.dtype:
                data = data.to(dtype=set_tensor.weight.dtype)
            set_tensor.weight.data = data
            return set_tensor.weight.data

        return func

    def _func_generator_set_module(value):
        def func(self, **kwargs):
            return _get_dst_obj(self, value, **kwargs).data.copy_(kwargs.get('data'))
        return func

    def _func_generator_set_bias(value):
        def func(self, **kwargs):
            set_tensor = _get_dst_obj(self, value, **kwargs)
            data = kwargs.get('data')
            if data.dtype != set_tensor.weight.dtype:
                data = data.to(dtype=set_tensor.weight.dtype)
            set_tensor.bias.data = data
            return set_tensor.bias.data

        return func

    def _func_generator_has_module(value):
        def func(self, **kwargs):
            obj = _get_src_obj(self, value, **kwargs)
            return True if obj else False
        return func

    def _func_generator_has_bias(value):
        def func(self, **kwargs):
            bias = getattr(_get_src_obj(self, value, **kwargs), 'bias', None)
            return bias is not None
        return func

    if self.module_mapping:
        for key, value in self.module_mapping.items():
            setattr(self, "get_" + key + "_module", _func_generator_get_module(value).__get__(self, ModelBase))
            setattr(self, "set_" + key + "_module", _func_generator_set_module(value).__get__(self, ModelBase))
            setattr(self, "get_" + key + "_weight", _func_generator_get_weight(value).__get__(self, ModelBase))
            setattr(self, "get_" + key + "_bias", _func_generator_get_bias(value).__get__(self, ModelBase))
            setattr(self, "set_" + key + "_weight", _func_generator_set_weight(value).__get__(self, ModelBase))
            setattr(self, "set_" + key + "_bias", _func_generator_set_bias(value).__get__(self, ModelBase))
            setattr(self, "has_" + key + "_module", _func_generator_has_module(value).__get__(self, ModelBase))
            setattr(self, "has_" + key + "_bias", _func_generator_has_bias(value).__get__(self, ModelBase))


def get_modules_from_pretrained(self, device_map="cpu", trust_remote_code=True):
    # Load Huggingface model.
    if self.args_cmd.save_model_type == "hf":
        load_dir = self.args_cmd.save_dir
    else:
        load_dir = self.args_cmd.load_dir
    
    self.module = [AutoModelForCausalLM.from_pretrained(
        load_dir, device_map=device_map, trust_remote_code=trust_remote_code, local_files_only=True, low_cpu_mem_usage=False
    )]

    if self.args_cmd.save_lora_to_hf:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.args_cmd.lora_r,
            lora_alpha=self.args_cmd.lora_alpha,
            target_modules=self.target_lora_modules_hf,
            lora_dropout=0.0,
            bias="none"
        )
        self.module = [get_peft_model(self.module[0], lora_config)]
    if hasattr(self.args, "torch_dtype") and self.args.torch_dtype in ["float16", "bfloat16"]:
        self.module[0] = self.module[0].to(eval(f'torch.{self.args.torch_dtype}'))
