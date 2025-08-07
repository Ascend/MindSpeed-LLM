# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from mindspeed.features_manager.feature import MindSpeedFeature


class AffinityFeature(MindSpeedFeature):
    def __init__(self):
        super(AffinityFeature, self).__init__('affinity', optimization_level=0)

    def register_patches(self, patch_manager, args):
        from mindspeed_llm.core.tensor_parallel.cross_entropy import calculate_predicted_logits
        # use logical negation followed by multiplication to achieve the same effect as setting selected elements to zero
        patch_manager.register_patch('megatron.core.tensor_parallel.cross_entropy.VocabParallelCrossEntropy.calculate_predicted_logits',
                                      calculate_predicted_logits)