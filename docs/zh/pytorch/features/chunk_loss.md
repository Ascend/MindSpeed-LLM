# ChunkLoss

## 背景与挑战

通过fsdp2训练模型时，`lm_head` 的输出维度（即词表大小 `vocab_size`）通常远大于模型的隐空间维度 `hidden_size`，传统损失计算方式需要在中间显式构造一个形状为 `[bs, seq, vocab_size]` 的 logits 张量，这会带来显著的显存尖刺，影响显存的利用率。

## 解决方案

通过对序列维度进行分块（chunking），将 loss 计算拆分为多个长度为`sub_seq`的子段依次进行。在完成每个子段的前向计算后，立即执行对应的反向传播，从而避免同时保留整个序列的 logits。这样一来，任意时刻最多只需缓存长度为 `sub_seq` 的 logits，显著降低了显存峰值。

## 使用方法

**第1步** 替换模型的lm_head(output_layer)实现（原实现为nn.Linear)
当前所有模型的lm_head都是没有bias的线性层，可改为以下实现
```python
class LMHead(nn.Linear):
    def forward(self, hidden_states: torch.Tensor, loss_ctx: callable = None):
        # Handle distributed tensor (DTensor) weights and biases by converting to local tensors.
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
**第2步**在模型的forward函数里添加loss_ctx入参，并在forward实现里添加使能判断

参考fsdp2 Qwen3ForCausalLM实现，[参考链接](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/mindspeed_llm/fsdp2/models/qwen3/qwen3.py)
此外，具体模型需要注意loss计算方式，如有新的loss计算方式，应在 build_loss_ctx 里适配修改，[修改位置在FSDP2Model的build_loss_ctx方法](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/mindspeed_llm/fsdp2/core/models/fsdp2_model.py)

**第3步**在启动脚本中添加使能参数
```shell
   --loss-compute-mode  chunk \
   --loss-chunk-size 1024 \
```

## 使用效果

启用 ChunkLoss 特性后，通过合理设置 `chunk_size`，可在显著降低显存峰值的同时保持相同的损失曲线。
