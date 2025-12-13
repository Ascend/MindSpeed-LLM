import transformers

from mindspeed_llm.fsdp2.core.fully_shard.fsdp2_sharding import FSDP2ShardingMixin
from mindspeed_llm.fsdp2.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM as GptOssWithFusionOps


class GptOssFSDP2Mixin(FSDP2ShardingMixin):
    """
    Mixin class for FSDP2 of the GPT-OSS-series
    """
    pass


class GptOssForCausalLM(GptOssWithFusionOps, GptOssFSDP2Mixin):
    @staticmethod
    def register_patches(config):
        """patching the transformers model."""
        pass