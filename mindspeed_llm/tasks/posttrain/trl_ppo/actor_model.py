from torch import Tensor
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron.core import InferenceParams
from megatron.core.distributed import DistributedDataParallel as LocalDDP
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import Float16Module as MegatronFloat16Module
from megatron.training import get_args
from megatron.training.utils import unwrap_model
from mindspeed_llm.inference.text_generation.tokenization import tokenize_prompts
from mindspeed_llm.tasks.inference import MegatronModuleForCausalLM


class ActorModel(MegatronModuleForCausalLM):
    """
    ActorModel supports both train and inference process. 
    The type of ActorModel.model.model is GPTModel
    """

    def __init__(self):
        super(ActorModel, self).__init__()
        self.model = None
        self.truncate = True

    def _post_processing(self, output, context_lengths, log_probs):
        input = [val[:context_lengths[i]] for i, val in enumerate(output)]

        if not self.include_input:
            output = [val[context_lengths[i]:] for i, val in enumerate(output)]

        # When batch size > 1, you need truncate the tokens after eos_token_id
        output = self._truncate_in_multi_batch(output)

        if not self.return_output_log_probs:
            res = output
        else:
            if self.num_beams == 1:
                log_probs = [val[context_lengths[i] - 1:, :] for i, val in enumerate(log_probs)] \
                    if log_probs is not None else None

            res = output, log_probs[0] if len(log_probs) == 1 else log_probs

        return input, res, context_lengths

    def generate(self, input_ids=None, **kwargs):
        args = get_args()

        super(MegatronModuleForCausalLM, self).generate(input_ids=input_ids, **kwargs)

        # =======================================
        # Add additional parameters to args which
        # may be used in original logic of codes
        # =======================================
        for addition_key, addition_val in kwargs.items():
            setattr(args, addition_key, addition_val)

        # =======================================
        # Initialize the tokenizer to choose
        # whether to use customizing tokenizer
        # =======================================
        self._init_tokenizer(args)
        args.pad_token_id = self.tokenizer.pad_token_id

        # =======================================
        # Tokenize the prompts
        # =======================================
        context_tokens_tensor, context_length_tensor = tokenize_prompts(tokenizer=self.tokenizer, prompts=input_ids, tokens_to_generate=self.max_new_tokens,
                                                                        max_generate_length=self.max_length, add_BOS=False, broadcast=False)

        args.seq_length = context_tokens_tensor.shape[1]
        args.max_position_embeddings = args.seq_length

        # =======================================
        # Get the streaming tokens generator
        # =======================================
        unwrap_classes = (torchDDP, LocalDDP, MegatronFloat16Module)
        generate_model = unwrap_model(self.model, unwrap_classes)[0]
        token_stream = self.greedy_search_or_sampling(
            generate_model,
            tokens=context_tokens_tensor,
            lengths=context_length_tensor,
            do_sample=self.do_sample,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
            return_output_log_probs=self.return_output_log_probs
        )

        # =======================================
        # Post processions in order to get final
        # output texts/tokens
        # =======================================
        return self._token_generator(token_stream)

    def forward(
            self,
            input_ids: Tensor,
            position_ids: Tensor,
            attention_mask: Tensor,
            decoder_input: Tensor = None,
            labels: Tensor = None,
            inference_params: InferenceParams = None,
            packed_seq_params: PackedSeqParams = None,
            extra_block_kwargs: dict = None,
    ):
        self.model.forward(input_ids,
                           position_ids,
                           attention_mask,
                           decoder_input,
                           labels,
                           inference_params,
                           packed_seq_params,
                           extra_block_kwargs)