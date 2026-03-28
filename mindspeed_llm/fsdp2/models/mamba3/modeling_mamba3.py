import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import distribute_tensor

from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple

from .mamba3_block import Mamba3


class Mamba2Config(PretrainedConfig):
    model_type = "mamba2"

    num_heads: int = 128
    head_dim: int = 64
    vocab_size: int = 32768
    hidden_size: int = 4096
    state_size: int = 128
    num_hidden_layers: int = 64
    layer_norm_epsilon: float = 1e-5
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | None = 2
    expand: int = 2
    conv_kernel: int = 4
    n_groups: int = 8
    use_bias: bool = False
    use_conv_bias: bool = True
    hidden_act: str = "silu"
    initializer_range: float = 0.1
    residual_in_fp32: bool = True
    time_step_rank: str | int = "auto"
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    time_step_limit: list[float] | tuple[float, ...] = (0.0, float("inf"))
    rescale_prenorm_residual: bool = False
    use_cache: bool = True
    rms_norm: bool = True
    chunk_size: int = 256
    tie_word_embeddings: bool = False


class Mamba3Config(Mamba2Config):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = self.hidden_size
        self.n_layer = self.num_hidden_layers
        self.d_state = self.state_size
        self.headdim = self.head_dim
        self.rms_norm_eps = self.layer_norm_epsilon


class MambaRMSNormGated(torch.nn.Module):
    def init(self, hidden_size, eps=1e-6):
        super().init()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        if gate is not None:
            hidden_states = hidden_states * nn.functional.silu(gate.to(torch.float32))
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states.to(input_dtype)


class Mamba3RMSNorm(nn.Module):
    def init(self, hidden_size, eps=1e-6):
        """
        Mamba3RMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm
        """
        super().init()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Mamba3Block(nn.Module):
    def init(self, config: Mamba3Config, layer_idx: int):
        super().init()

        self.mamba = Mamba3(
            d_model=config.hidden_size, is_mimo=False
        )

    def forward(self, hidden_states, inference_params=None):
        residual = hidden_states
        hidden_states = self.mamba(hidden_states)
        return residual + hidden_states


class Mamba3Model(PreTrainedModel):
    config_class = Mamba3Config

    def __init__(self, config: Mamba3Config):
        super().__init__(config)

        vocab_size = max(config.vocab_size, 200000)
        self.embeddings = nn.Embedding(vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([
            Mamba3Block(config, layer_idx=i)
            for i in range(config.n_layer)
        ])
        self.norm_f = Mamba3RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)


    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def forward(
        self,
        input_ids=None,
        inference_params=None,
        **kwargs,
    ):

        inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

        for layer in self.layers:
            hidden_states = layer(hidden_states, inference_params=inference_params)

        hidden_states = self.norm_f(hidden_states)
        return hidden_states


    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Mamba3Block):
            dt = torch.exp(
                torch.rand(module.mamba.nheads) * (math.log(module.mamba.dt_max) - math.log(module.mamba.dt_min))
                + math.log(module.mamba.dt_min)
            )
            dt = torch.clamp(dt, min=module.mamba.dt_init_floor)
            dt_bias = dt + torch.log(-torch.expm1(-dt))

            dt_bias = distribute_tensor(
                    dt_bias,
                    module.mamba.dt_bias.device_mesh,
                    module.mamba.dt_bias.placements
                )
            module.mamba.dt_bias[:] = dt_bias
            module.mamba.B_bias.data.fill_(1.0)
            module.mamba.C_bias.data.fill_(1.0)
            if module.mamba.is_mimo:
                module.mamba.mimo_x.data.fill_(1.0)
                module.mamba.mimo_z.data.fill_(1.0)
                module.mamba.mimo_o.data.fill_(1.0)
            module.mamba.D.data.fill_(1.0)


class Mamba2ForCausalLM(PreTrainedModel):
    config_class = Mamba2Config

    def __init__(self, config: Mamba2Config):
        super().__init__(config)


class Mamba3ForCausalLM(Mamba2ForCausalLM):
    config_class = Mamba3Config

    def __init__(self, config: Mamba3Config):
        super().__init__(config)

        self.model = Mamba3Model(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)


    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.model.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels=None,
        inference_params=None,
        attention_mask: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ):
        mamba3_outputs = self.model(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        hidden_states = mamba3_outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
        )