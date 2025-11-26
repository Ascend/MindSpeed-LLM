import transformers

from mindspeed_llm.fsdp2.core.fully_shard.fsdp2_sharding import FSDP2ShardingMixin


class GptOssFSDP2Mixin(FSDP2ShardingMixin):
    """
    Mixin class for FSDP2 of the GPT-OSS-series
    """
    pass


class GptOssForCausalLM(transformers.GptOssForCausalLM, GptOssFSDP2Mixin):
    pass