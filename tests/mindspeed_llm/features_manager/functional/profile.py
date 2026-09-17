# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
from argparse import ArgumentParser

from megatron_adaptor.features_manager import ProfileFeature as MAProfileFeature


class ProfileFeature(MAProfileFeature):

    def register_patches(self, patch_manager, args):
        super().register_patches(patch_manager, args)

        from megatron_adaptor.patches.megatron.profile import train_wrapper

        patch_manager.register_patch('mindspeed_llm.training.training.train', train_wrapper)