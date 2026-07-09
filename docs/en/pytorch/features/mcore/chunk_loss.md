# ChunkLoss

## Background and Challenges

When you train a model with FSDP2, the output dimension of `lm_head`, that is, the vocabulary size `vocab_size`, is usually much larger than the hidden size of the model, `hidden_size`. The traditional loss calculation method needs to explicitly construct a logits tensor with the shape `[bs, seq, vocab_size]` in the middle of the process. This creates a significant memory spike and lowers memory utilization.

## Solution

By chunking the sequence dimension, you split loss calculation into multiple subsequences of length `sub_seq` and process them one by one. After each subsequence finishes its forward pass, immediately run the corresponding backward pass. This avoids keeping the logits for the entire sequence at the same time. As a result, you only need to cache logits for at most `sub_seq` tokens at any moment, which significantly lowers peak memory use.

## Usage

**Step 1**: Replace the `lm_head` (`output_layer`) implementation of the model. The original implementation uses `nn.Linear`.

All current models use a bias-free linear layer for `lm_head`, and you can replace it with the following implementation.

```python
class LMHead(nn.Linear):
    def forward(self, hidden_states: torch.Tensor, loss_ctx: callable = None):
        # Handle distributed tensor (DTensor) weights and biases by converting them to local tensors.
        if isinstance(self.weight, DTensor):
            w = self.weight.to_local()
            if self.bias is not None:
                if not isinstance(self.bias, DTensor):
                    raise TypeError(
                        f"Expected bias to be a DTensor when weight is a DTensor, "
                        f"but got bias of type {type(self.bias)}."
                    )
                b = self.bias.to_local()
            else:
                b = None
        else:
            w = self.weight
            b = self.bias

        if loss_ctx is None:
            # If no loss context is provided, compute and return logits normally.
            logits = F.linear(hidden_states, w, b)
            return logits, None
        else:
            # Otherwise, delegate loss computation to the provided loss context function,
            # which typically enables memory-efficient or chunked loss calculation.
            return None, loss_ctx(hidden_states, w, b)
```

**Step 2**: Add the `loss_ctx` argument to the `forward` function of the model, and add an enablement check in the forward implementation.

See the FSDP2 `Qwen3ForCausalLM` implementation: [link](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/mindspeed_llm/fsdp2/models/qwen3/qwen3.py).
In addition, pay attention to the loss calculation method of the specific model. If you introduce a new loss calculation method, adapt the `_build_chunk_loss` logic accordingly. The change point is the Trainer `_build_chunk_loss` method, [modification location](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/mindspeed_llm/fsdp2/train/trainer.py#L86).

**Step 3**: Add the enablement arguments to the launch script.

```shell
   --loss-compute-mode  chunk \
   --loss-chunk-size 1024 \
```

## Expected Result

After enabling the ChunkLoss feature, you can set `chunk_size` appropriately to significantly reduce peak memory use while keeping the same loss curve.
