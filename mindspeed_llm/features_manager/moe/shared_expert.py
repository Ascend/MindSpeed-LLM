from argparse import ArgumentParser

from mindspeed.features_manager.moe.shared_expert import MoESharedExpertsFeature as MindSpeedMoESharedExpertsFeature


class MoESharedExpertsFeature(MindSpeedMoESharedExpertsFeature):
    def pre_validate_args(self, args):
        # use megatron shared_experts replace
        if args.n_shared_experts and args.moe_shared_expert_intermediate_size is None:
            args.moe_shared_expert_intermediate_size = args.n_shared_experts * args.moe_ffn_hidden_size